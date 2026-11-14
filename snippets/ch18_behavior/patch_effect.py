def patch_effect(recipient_gap, donor_gap, patched_gap):
    """Return the signed shift and its fraction of the donor-recipient gap."""
    shift = patched_gap - recipient_gap
    span = donor_gap - recipient_gap
    recovery = shift / span if abs(span) > 1e-8 else None
    return shift, recovery
