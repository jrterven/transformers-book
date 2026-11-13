"""Measure image-text retrieval and visual answers from released checkpoints."""

import argparse
from importlib.metadata import version
from pathlib import Path
import json
import re

from huggingface_hub import hf_hub_download, snapshot_download
import numpy as np
from PIL import ImageOps
import torch
from torch.nn import functional as F
from transformers import AutoProcessor, Qwen3VLModel, Qwen3VLProcessor, Qwen3_5ForConditionalGeneration, SiglipModel

from .demo_data import ATTRIBUTES, CAPTIONS, PAIRS, RELATION_CAPTIONS, load_photos, photo_manifest, verify_file
from .model import bidirectional_recall_at_k, contrastive_logits


SIGLIP = {
    "repo": "google/siglip2-base-patch16-224", "revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    "weights": "model.safetensors", "sha256": "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b",
}
EMBEDDER = {
    "repo": "Qwen/Qwen3-VL-Embedding-2B", "revision": "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda",
    "weights": "model.safetensors", "sha256": "c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1",
}
ASSISTANT = {
    "repo": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "weights": "model.safetensors-00001-of-00001.safetensors",
    "sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
}
INSTRUCTION = "Represent the user's input."
MAX_NEW_TOKENS = 12


def checkpoint(spec, offline):
    arguments = {"revision": spec["revision"], "local_files_only": offline}
    weights = Path(hf_hub_download(spec["repo"], spec["weights"], **arguments))
    verify_file(weights, spec["sha256"])
    if not offline:
        snapshot_download(spec["repo"], **arguments, allow_patterns=["*.json", "*.jinja", "*.txt", "*.model"])
    return weights.parent


def load_model(cls, path, dtype):
    model, report = cls.from_pretrained(
        path, dtype=dtype, attn_implementation="sdpa", output_loading_info=True,
        use_safetensors=True, local_files_only=True, trust_remote_code=False,
    )
    if any(report.values()):
        raise RuntimeError(f"Incomplete checkpoint load: {report}")
    return model.eval()


def shapes(inputs):
    return {key: list(value.shape) for key, value in inputs.items() if torch.is_tensor(value)}


def model_record(spec, model):
    return {**spec, "parameters": sum(p.numel() for p in model.parameters()),
            "dtype": str(model.dtype).removeprefix("torch.")}


def retrieval_results(scores, relevance):
    ranked = {}
    for name, rows in (("image_to_text", scores), ("text_to_image", scores.T)):
        ranked[name] = np.argsort(-rows, axis=1, kind="stable").tolist()
    metrics = {}
    for k in (1, 2, 3):
        result = bidirectional_recall_at_k(scores, relevance, k)
        for side, direction in (("image_to_text", relevance), ("text_to_image", relevance.T)):
            order = np.asarray(ranked[side])[:, :k]
            hits = np.take_along_axis(direction, order, axis=1).any(axis=1)
            defined = direction.any(axis=1)
            result[side]["hits"] = int(hits[defined].sum())
        metrics[str(k)] = result
    return {"cosines": scores.tolist(), "ranked": ranked, "metrics": metrics}


def relation_probe(photos, score):
    rows = []
    for key, photo, _, expected in PAIRS:
        if key not in RELATION_CAPTIONS:
            continue
        images = [photos[photo], ImageOps.mirror(photos[photo])]
        captions = [RELATION_CAPTIONS[key].format(direction) for direction in ("left", "right")]
        similarities = score(images, captions)
        choices = [None if left == right else ("right" if right > left else "left")
                   for left, right in similarities]
        rows.append({"id": key, "photo": photo, "captions_left_right": captions,
                     "cosines_original_mirror": similarities.tolist(), "winner_original_mirror": choices,
                     "expected_original_mirror": list(expected)})
    return rows


@torch.inference_mode()
def siglip_retrieval(photos, relevance, offline):
    path = checkpoint(SIGLIP, offline)
    model = load_model(SiglipModel, path, torch.float32)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)

    def encode(images, captions):
        batch = processor(images=images, text=captions, padding="max_length", return_tensors="pt")
        return batch, model(**batch)

    inputs, output = encode(list(photos.values()), [text for _, text, _ in CAPTIONS])
    image, text = output.image_embeds, output.text_embeds
    scale, bias = model.logit_scale.exp().item(), model.logit_bias.item()
    scores = contrastive_logits(image.numpy(), text.numpy(), scale=1.0)
    np.testing.assert_allclose(scale * scores + bias, output.logits_per_image.numpy(), atol=2e-5)

    def score(images, captions):
        _, encoded = encode(images, captions)
        return contrastive_logits(encoded.image_embeds.numpy(), encoded.text_embeds.numpy(), scale=1.0)

    return {"model": model_record(SIGLIP, model), "inputs": shapes(inputs), "width": image.shape[-1],
            "logit_scale": scale, "logit_bias": bias, "logits": output.logits_per_image.tolist(),
            "pair_sigmoids": output.logits_per_image.sigmoid().tolist(), "relation_probe": relation_probe(photos, score),
            **retrieval_results(scores, relevance)}


def conversation(content, instruction=None):
    messages = [{"role": "user", "content": content}]
    if instruction:
        messages.insert(0, {"role": "system", "content": [{"type": "text", "text": instruction}]})
    return messages


def image_message(photo):
    return [{"type": "image", "image": photo}]


def text_message(text):
    return [{"type": "text", "text": text}]


@torch.inference_mode()
def embedding_retrieval(photos, relevance, offline):
    path = checkpoint(EMBEDDER, offline)
    model = load_model(Qwen3VLModel, path, torch.bfloat16)
    processor = Qwen3VLProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)

    def encode(content):
        inputs = processor.apply_chat_template(
            conversation(content, INSTRUCTION), add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        output = model(**inputs, use_cache=False)
        last = inputs.attention_mask.shape[1] - 1 - inputs.attention_mask.flip(1).argmax(dim=1)
        pooled = output.last_hidden_state[torch.arange(len(last)), last].float()
        return F.normalize(pooled, dim=-1)[0], int(last[0]), int(inputs.input_ids[0, last[0]])

    def encode_images(images):
        return [encode(image_message(photo)) for photo in images]

    def encode_captions(captions):
        return [encode(text_message(text)) for text in captions]

    images = encode_images(photos.values())
    captions = encode_captions(text for _, text, _ in CAPTIONS)
    image_rows, text_rows = (torch.stack([value[0] for value in rows]) for rows in (images, captions))
    scores = contrastive_logits(image_rows.numpy(), text_rows.numpy(), scale=1.0)

    def score(images, captions):
        image_vectors = torch.stack([item[0] for item in encode_images(images)])
        text_vectors = torch.stack([item[0] for item in encode_captions(captions)])
        return contrastive_logits(image_vectors.numpy(), text_vectors.numpy(), scale=1.0)

    return {"model": model_record(EMBEDDER, model), "width": image_rows.shape[-1],
            "system_instruction": INSTRUCTION, "image_last_position": [row[1] for row in images],
            "text_last_position": [row[1] for row in captions],
            "pooled_token_ids": [row[2] for row in images + captions], "relation_probe": relation_probe(photos, score),
            **retrieval_results(scores, relevance)}


def answer_key(raw):
    return re.sub(r"[.!?]+$", "", raw.strip().lower())


@torch.inference_mode()
def visual_questions(photos, offline):
    path = checkpoint(ASSISTANT, offline)
    model = load_model(Qwen3_5ForConditionalGeneration, path, torch.float32)
    processor = Qwen3VLProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    direction_tokens = {label: processor.tokenizer.encode(" " + label, add_special_tokens=False)
                        for label in ("left", "right")}
    if any(len(ids) != 1 for ids in direction_tokens.values()):
        raise ValueError("The two direction alternatives should each encode as one token")
    left_id, right_id = (direction_tokens[label][0] for label in ("left", "right"))

    def ask(question, photo=None, *, direction=False, inspect=False):
        content = (image_message(photo) if photo is not None else []) + text_message(question)
        inputs = processor.apply_chat_template(
            conversation(content), add_generation_prompt=True, enable_thinking=False,
            tokenize=True, return_dict=True, return_tensors="pt",
        )
        n = inputs.input_ids.shape[1]
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                                   return_dict_in_generate=True, output_scores=True)
        continuation = generated.sequences[0, n:]
        raw = processor.tokenizer.decode(continuation, skip_special_tokens=True)
        record = {"answer": raw.strip(), "normalized": answer_key(raw), "ids": continuation.tolist(),
                  "input_tokens": n, "image_tokens": int((inputs.input_ids == model.config.image_token_id).sum())}
        if direction:
            scores = generated.scores[0][0].float()
            alternatives = scores[[left_id, right_id]]
            record.update({"left_right_logits": alternatives.tolist(),
                           "right_share_of_two": alternatives.softmax(0)[1].item(),
                           "right_token_full_vocab_probability": scores.softmax(0)[right_id].item()})
        if inspect:
            visual = model.get_image_features(inputs.pixel_values, inputs.image_grid_thw, return_dict=True)
            record["trace"] = {
                "image_grid_thw": inputs.image_grid_thw[0].tolist(), "pixels": list(inputs.pixel_values.shape),
                "encoded_patches": list(visual.last_hidden_state.shape), "merged_tokens": list(visual.pooler_output[0].shape),
                "patch_size": model.config.vision_config.patch_size,
                "spatial_merge_size": model.config.vision_config.spatial_merge_size,
                "temporal_patch_size": model.config.vision_config.temporal_patch_size,
                "vocabulary": model.config.text_config.vocab_size,
            }
            if record["image_tokens"] != visual.pooler_output[0].shape[0]:
                raise ValueError("Number of image placeholders differs from the merged visual rows")
        return record

    pairs = []
    for key, photo, question, answers in PAIRS:
        original = ask(question, photos[photo], direction=True, inspect=key == PAIRS[0][0])
        mirror = ask(question, ImageOps.mirror(photos[photo]), direction=True)
        question_only = ask(question, direction=True)
        pairs.append({"id": key, "photo": photo, "question": question, "expected": list(answers),
                      "original": original, "mirror": mirror, "question_only": question_only})
    attributes = [{"id": key, "photo": photo, "question": question, "accepted": list(accepted),
                   **ask(question, photos[photo])} for key, photo, question, accepted in ATTRIBUTES]

    metrics = {}
    for name, views in (("visual", ("original", "mirror")), ("text_only", ("question_only", "question_only"))):
        outputs = [[row[view]["normalized"] for view in views] for row in pairs]
        correct = [[guess == target for guess, target in zip(pair, row["expected"], strict=True)]
                   for pair, row in zip(outputs, pairs, strict=True)]
        metrics[name] = {"correct_views": sum(sum(row) for row in correct), "views": 2 * len(pairs),
                         "both_correct": sum(all(row) for row in correct), "pairs": len(pairs),
                         "different_answers": sum(first != second for first, second in outputs)}
    return {"model": model_record(ASSISTANT, model), "thinking": False, "decode": "greedy",
            "max_new_tokens": MAX_NEW_TOKENS, "direction_tokens": direction_tokens,
            "pairs": pairs, "metrics": metrics, "attributes": attributes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/ch23-pretrained.json"))
    parser.add_argument("--media-dir", type=Path, default=Path("output/ch23-media"))
    parser.add_argument("--offline", action="store_true", help="Require cached media and checkpoints")
    parser.add_argument("--experiment", choices=("all", "siglip", "embedding", "visual"), default="all")
    args = parser.parse_args()
    torch.manual_seed(23)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    photos = load_photos(args.media_dir, args.offline)
    relevance = np.array([[key in relevant for _, _, relevant in CAPTIONS] for key in photos])
    record = {"settings": {"device": "cpu", "seed": 23, "threads": 8,
                           "packages": {key: version(key) for key in (
                               "numpy", "torch", "torchvision", "transformers", "huggingface-hub", "pillow",
                           )}},
              "photos": photo_manifest(photos), "image_order": list(photos),
              "captions": [{"id": key, "text": text, "relevant": list(matches)} for key, text, matches in CAPTIONS],
              "relevance": relevance.tolist()}
    if args.experiment in ("all", "siglip"):
        record["siglip"] = siglip_retrieval(photos, relevance, args.offline)
    if args.experiment in ("all", "embedding"):
        record["embedding"] = embedding_retrieval(photos, relevance, args.offline)
    if args.experiment in ("all", "visual"):
        record["visual"] = visual_questions(photos, args.offline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved the {args.experiment} observations to {args.output}")


if __name__ == "__main__":
    main()
