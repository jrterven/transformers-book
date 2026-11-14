"""Measure boxes and masks from released RF-DETR and SAM 2.1 checkpoints."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

from huggingface_hub import snapshot_download
import numpy as np
import torch
from torch.nn import functional as F
from transformers import (
    AutoImageProcessor,
    RfDetrForInstanceSegmentation,
    RfDetrForObjectDetection,
    Sam2Config,
    Sam2Model,
    Sam2Processor,
)

from .demo_data import encode_mask, load_photo, overlap_counts, read_fixture, reference_instances, verify_file
from .model import all_points_ap, box_iou, precision_recall


DETECTOR = {
    "repo": "Roboflow/rf-detr-nano",
    "revision": "9f201577d9415f0378084748d27e68586e6b3600",
    "sha256": "497b1dbcc8f9d115b806fb82d140d7e22e5b5c3fba45abefef39dd4f99c1f737",
}
INSTANCE_MODEL = {
    "repo": "Roboflow/rf-detr-seg-nano",
    "revision": "472badbc863609789ce1f69311244c670bdfd2b9",
    "sha256": "769bf8c0d180f3a4d0dc7fedfbfe561023d1b00754e669b6aac16d713b8c4050",
}
PROMPT_MODEL = {
    "repo": "facebook/sam2.1-hiera-tiny",
    "revision": "de431c4043854a71d8101e17995dfe596bf101a5",
    "sha256": "48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7",
}
DISPLAY_CUTOFF = 0.5
RANKED_CUTOFF = 0.04
IOU_THRESHOLDS = (0.50, 0.75, 0.90, 0.93, 0.95)
SHIFT_X = 96


def checkpoint(spec, offline):
    path = Path(snapshot_download(
        spec["repo"], revision=spec["revision"],
        allow_patterns=["*.json", "model.safetensors"], local_files_only=offline,
    ))
    verify_file(path / "model.safetensors", spec["sha256"])
    return path


def load_model(model_class, snapshot, **kwargs):
    model, report = model_class.from_pretrained(
        snapshot, use_safetensors=True, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, output_loading_info=True, **kwargs,
    )
    if any(report.values()):
        raise RuntimeError(f"Incomplete checkpoint load: {report}")
    return model.eval()


def model_record(spec, model):
    return {**spec, "parameters": sum(parameter.numel() for parameter in model.parameters())}


def decode_rf_detr(outputs, processor, image, cutoff):
    # The processor ranks query/class pairs, so retain both indices in the trace.
    width, height = image.size
    logits = outputs.logits[0]
    pairs = logits.sigmoid().flatten().topk(logits.shape[0])
    actual = processor.post_process_object_detection(
        outputs, threshold=cutoff, target_sizes=[(height, width)],
    )[0]
    indices = pairs.indices[pairs.values > cutoff]
    queries, classes = indices // logits.shape[1], indices % logits.shape[1]
    torch.testing.assert_close(classes, actual["labels"])
    boxes = outputs.pred_boxes[0, queries]
    corners = torch.cat([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], dim=-1)
    corners *= corners.new_tensor([width, height, width, height])
    torch.testing.assert_close(corners, actual["boxes"])
    torch.testing.assert_close(logits[queries, classes].sigmoid(), actual["scores"])
    return [
        {"query": q, "category_id": c, "logit": logits[q, c].item(),
         "score": score.item(), "box_cxcywh": box.tolist(), "box_xyxy": edge.tolist()}
        for q, c, score, box, edge in zip(
            queries.tolist(), classes.tolist(), actual["scores"], boxes, corners, strict=True,
        )
    ]


def rank_boxes(ground_truth, candidates):
    reference = [row["box_xyxy"] for row in ground_truth]
    predictions = [row["box_xyxy"] for row in candidates]
    scores = [row["score"] for row in candidates]
    evaluations = {}
    for threshold in IOU_THRESHOLDS:
        order, matches, precision, recall = precision_recall(reference, predictions, scores, threshold)
        if order.tolist() != list(range(len(candidates))):
            raise ValueError("The detector candidates must be in descending score order")
        evaluations[f"{threshold:.2f}"] = {
            "true_positive": matches.tolist(), "precision": precision.tolist(), "recall": recall.tolist(),
            "ap": all_points_ap(precision, recall),
        }
    return evaluations


@torch.inference_mode()
def detect(image, targets, offline):
    snapshot = checkpoint(DETECTOR, offline)
    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model = load_model(RfDetrForObjectDetection, snapshot)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    results = decode_rf_detr(outputs, processor, image, RANKED_CUTOFF)
    for row in results:
        row["label"] = model.config.id2label[row["category_id"]]
        references = targets.get(row["label"], ())
        row["box_ious"] = [box_iou(row["box_xyxy"], reference["box_xyxy"]) for reference in references]
    display = [row for row in results if row["score"] > DISPLAY_CUTOFF and row["label"] in targets]
    return {
        "model": model_record(DETECTOR, model),
        "input_shape": list(inputs.pixel_values.shape),
        "logit_shape": list(outputs.logits.shape), "box_shape": list(outputs.pred_boxes.shape),
        "display": display, "ranked": results,
        "evaluation": {name: rank_boxes(references, [row for row in results if row["label"] == name])
                       for name, references in targets.items()},
    }


@torch.inference_mode()
def segment_instances(image, targets, offline):
    snapshot = checkpoint(INSTANCE_MODEL, offline)
    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model = load_model(RfDetrForInstanceSegmentation, snapshot)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    boxes = decode_rf_detr(outputs, processor, image, DISPLAY_CUTOFF)
    target_size = image.size[::-1]
    official = processor.post_process_instance_segmentation(
        outputs, threshold=DISPLAY_CUTOFF, mask_threshold=0, target_sizes=[target_size],
        return_binary_maps=True,
    )[0]
    query_ids = [row["query"] for row in boxes]
    continuous = F.interpolate(
        outputs.pred_masks[0, query_ids, None].float(), target_size,
        mode="bilinear", align_corners=False,
    )[:, 0]
    masks = continuous > 0
    nonempty = masks.flatten(1).any(dim=1).tolist()
    torch.testing.assert_close(masks[nonempty], official["segmentation"])
    results = []
    for row, mask, keep in zip(boxes, masks, nonempty, strict=True):
        if not keep:
            continue
        label = model.config.id2label[row["category_id"]]
        measured = [overlap_counts(mask.numpy(), reference["mask"]) for reference in targets.get(label, ())]
        results.append({**row, "label": label, "mask": encode_mask(mask.numpy()), "mask_overlaps": measured})
    return {
        "model": model_record(INSTANCE_MODEL, model), "input_shape": list(inputs.pixel_values.shape),
        "logit_shape": list(outputs.logits.shape), "mask_logit_shape": list(outputs.pred_masks.shape),
        "instances": results,
    }


def prompt_rows(model, processor, embeddings, original_sizes, targets, **prompts):
    encoded = processor(original_sizes=original_sizes, return_tensors="pt", **prompts)
    args = {key: encoded[key] for key in ("input_points", "input_labels", "input_boxes") if key in encoded}
    outputs = model(image_embeddings=embeddings, **args, multimask_output=True)
    masks = processor.post_process_masks(outputs.pred_masks, original_sizes, mask_threshold=0)[0].numpy()
    quality = outputs.iou_scores[0].tolist()
    rows = []
    for candidates, scores, target in zip(masks, quality, targets, strict=True):
        measures = [{"quality": score, **overlap_counts(mask, target["mask"]), "mask": encode_mask(mask)}
                    for mask, score in zip(candidates, scores, strict=True)]
        selected = max(range(len(measures)), key=lambda index: scores[index])
        best_measured = max(range(len(measures)), key=lambda index: measures[index]["iou"])
        rows.append({"annotation_id": target["annotation_id"], "candidates": measures,
                     "selected_by_quality": selected, "best_measured": best_measured})
    return {"prompts": prompts, "encoded": {key: value.tolist() for key, value in args.items()},
            "raw_mask_shape": list(outputs.pred_masks.shape), "original_mask_shape": list(masks.shape),
            "rows": rows}


@torch.inference_mode()
def prompt_masks(image, cat_targets, detections, offline):
    snapshot = checkpoint(PROMPT_MODEL, offline)
    # Use the image architecture with the matching fields of the shared SAM 2.1 config.
    config = Sam2Config.from_dict(json.loads((snapshot / "config.json").read_text(encoding="utf-8")))
    processor = Sam2Processor.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    model = load_model(Sam2Model, snapshot, config=config)
    inputs = processor(images=image, return_tensors="pt")
    embeddings = model.get_image_embeddings(inputs.pixel_values)
    selected = sorted((row for row in detections if row["label"] == "cat"), key=lambda row: row["box_xyxy"][0])
    if len(selected) != len(cat_targets):
        raise ValueError("The fixed example expects two retained cat detections")
    boxes = [row["box_xyxy"] for row in selected]
    for index, (prediction, target) in enumerate(zip(selected, cat_targets, strict=True)):
        if prediction["box_ious"][index] < .5:
            raise ValueError(f"No matching detector prompt for annotation {target['annotation_id']}")
    shifted_left = [coord + SHIFT_X if i % 2 == 0 else coord for i, coord in enumerate(boxes[0])]
    point = [270, 209]
    if not cat_targets[0]["mask"][point[1], point[0]]:
        raise ValueError("The left foreground click must lie on the annotated cat")
    conditions = {
        "annotated_boxes": {"input_boxes": [[row["box_xyxy"] for row in cat_targets]]},
        "detector_boxes": {"input_boxes": [boxes]},
        "shifted_box": {"input_boxes": [[shifted_left, boxes[1]]]},
    }
    runs = {name: prompt_rows(model, processor, embeddings, inputs.original_sizes, cat_targets, **args)
            for name, args in conditions.items()}
    runs["foreground_click"] = prompt_rows(
        model, processor, embeddings, inputs.original_sizes, cat_targets[:1],
        input_points=[[[point]]], input_labels=[[[1]]],
    )
    return {"model": model_record(PROMPT_MODEL, model), "input_shape": list(inputs.pixel_values.shape),
            "embedding_shapes": [list(x.shape) for x in embeddings], "box_shift_x": SHIFT_X, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/ch23-pretrained.json"))
    parser.add_argument("--media-dir", type=Path, default=Path("output/ch23-media"))
    parser.add_argument("--offline", action="store_true", help="Require cached media and model weights")
    args = parser.parse_args()
    torch.manual_seed(22)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    fixture = read_fixture()
    image = load_photo(fixture, args.media_dir, args.offline)
    targets = reference_instances(fixture)
    detection = detect(image, targets, args.offline)
    instances = segment_instances(image, targets, args.offline)
    prompted = prompt_masks(image, targets["cat"], detection["display"], args.offline)
    data = {
        "settings": {"device": "cpu", "dtype": "float32", "seed": 22, "threads": 8,
                     "detection_cutoff": DISPLAY_CUTOFF, "ranking_cutoff": RANKED_CUTOFF,
                     "mask_logit_threshold": 0, "packages": {key: version(key) for key in (
                         "torch", "torchvision", "transformers", "numpy", "pillow", "pycocotools",
                     )}},
        "image": fixture["image"], "detection": detection, "instance_segmentation": instances,
        "prompted_segmentation": prompted,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved observations for three checkpoints to {args.output}")


if __name__ == "__main__":
    main()
