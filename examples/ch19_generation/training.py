"""Optional CPU step using the existing Chapter 5/6 encoder-decoder."""

import torch

from examples.ch05_transformer.model import Transformer
from examples.ch06_training.data import Batch
from examples.ch06_training.training import token_loss, train_step

from .model import collate_pairs


# snippet:encoder_decoder_step
def encoder_decoder_step(pairs):
    """Run one supervised update and measure loss on that same batch."""
    arrays = collate_pairs(pairs)
    batch = Batch(*(torch.from_numpy(array) for array in arrays))
    source_vocab = int(batch.source.max()) + 1
    target_vocab = max(int(batch.decoder_input.max()), int(batch.target.max())) + 1
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        model = Transformer(
            source_vocab, target_vocab, d_model=8, n_heads=2, d_ff=16,
            dropout=0, max_length=max(batch.source.shape[1], batch.target.shape[1]),
        ).double()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before = train_step(model, optimizer, batch)
    model.eval()
    with torch.no_grad():
        logits = model(batch.source, batch.decoder_input)
        total, count = token_loss(logits, batch.target)
    return {
        "logit_shape": list(logits.shape),
        "scored_tokens": int(count),
        "loss_before": before,
        "loss_after": float(total / count),
    }
# end-snippet:encoder_decoder_step
