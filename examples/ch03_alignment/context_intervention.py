"""Exchange one recurrent attention context between two measured decoder runs."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import torch

from examples.ch02_recurrence.data import BOS, VOCAB
from .inspect_trained import open_read, prefix_state
from .model import AttentiveSeq2Seq

SOURCE = ("blue", "blue", "yellow", "green", "white", "red")
CANDIDATES = ("green", "yellow")


# snippet:decode_with_context
def decode_with_context(model, previous, state, context):
    embedding = model.target_embedding(previous)
    hidden, _ = model.decoder(torch.cat((embedding, context), dim=-1), state)
    return model.output(hidden)
# end-snippet:decode_with_context


@torch.inference_mode()
def measure(checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["configuration"]
    model = AttentiveSeq2Seq(**{k: config[k] for k in
                               ("embedding_width", "hidden_width", "alignment_width")})
    model.load_state_dict(saved["model"])
    model.eval()
    memory, shared_state = prefix_state(model, [VOCAB.index(w) for w in SOURCE], [BOS])
    previous = torch.tensor([VOCAB.index("white")])
    candidates = [VOCAB.index(w) for w in CANDIDATES]
    states, weights, contexts = [], [], []
    for early_input in ("red", "blue"):
        _, state, _ = model.step(torch.tensor([VOCAB.index(early_input)]), shared_state, memory)
        _, row, _, context = open_read(model, state, memory)
        native, _, _ = model.step(previous, state, memory)
        # Reinserting a run's own context is a direct check of the intervention.
        torch.testing.assert_close(decode_with_context(model, previous, state, context), native)
        states.append(state)
        weights.append(row[0].tolist())
        contexts.append(context)

    conditions = []
    for state_id, context_id in ((0, 0), (1, 1), (1, 0), (0, 1)):
        logits = decode_with_context(model, previous, states[state_id], contexts[context_id])[0]
        p = logits.double().softmax(-1)
        conditions.append({
            "state": ("normal", "changed")[state_id],
            "context": ("normal", "changed")[context_id],
            "logits_green_yellow": logits[candidates].tolist(),
            "green": float(p[candidates[0]]), "yellow": float(p[candidates[1]]),
            "gap": float(logits[candidates[0]] - logits[candidates[1]]),
            "winner": VOCAB[int(logits.argmax())],
        })
    with checkpoint.open("rb") as f:
        digest = file_digest(f, "sha256").hexdigest()
    return {"source": SOURCE, "step": 3, "common_input": "white", "candidate_order": CANDIDATES,
            "checkpoint_sha256": digest, "torch": torch.__version__, "update": saved["selected_step"],
            "weights": weights, "context_first_two": [c[0, :2].tolist() for c in contexts],
            "conditions": conditions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = measure(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["conditions"], indent=2))


if __name__ == "__main__":
    main()
