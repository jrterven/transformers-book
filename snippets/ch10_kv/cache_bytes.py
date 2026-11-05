def cache_bytes(batch, layers, length, kv_heads, key_width, value_width,
                bytes_per_scalar):
    """Count the K/V array payload, assuming the same dtype in every layer."""
    batch, layers, length, kv_heads, key_width, value_width, bytes_per_scalar = map(
        index, (batch, layers, length, kv_heads, key_width, value_width,
                bytes_per_scalar)
    )
    if length < 0 or min(batch, layers, kv_heads, key_width, value_width,
                         bytes_per_scalar) < 1:
        raise ValueError("cache dimensions must be positive, except empty history")
    per_position = kv_heads * (key_width + value_width) * bytes_per_scalar
    return batch * layers * length * per_position
