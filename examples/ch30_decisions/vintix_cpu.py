"""Use the PyTorch causal attention kernel for Vintix II's CPU inference."""

from types import SimpleNamespace


def use_cpu_attention():
    from torch.nn.functional import scaled_dot_product_attention
    from vintix.model import model

    def causal_attention(qkv, dropout_p=0.0, causal=False):
        query, key, value = (x.transpose(1, 2) for x in qkv.unbind(dim=2))
        attention = scaled_dot_product_attention(
            query, key, value, dropout_p=dropout_p, is_causal=causal,
        )
        return attention.transpose(1, 2)

    # The pinned upstream forward pass recomputes its context with no KV cache.
    model.flash_attn = SimpleNamespace(flash_attn_qkvpacked_func=causal_attention)
