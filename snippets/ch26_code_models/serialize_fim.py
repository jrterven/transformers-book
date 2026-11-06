def serialize_fim(prefix, suffix):
    """Show a generic prefix-suffix-middle layout with teaching markers."""
    if not isinstance(prefix, str) or not isinstance(suffix, str):
        raise TypeError("prefix and suffix must be strings")
    return f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
