def quote_total(party_size, rate_cents, fee_cents):
    """Compute exact cents after the runner validates the integers."""
    return {"total_cents": party_size * rate_cents + fee_cents}
