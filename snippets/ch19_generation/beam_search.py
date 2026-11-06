def beam_search(next_probabilities, beam_width, max_steps, eos_id=EOS):
    """Return the best surviving tokens, log probability, and EOS status."""
    beam_width, max_steps = index(beam_width), index(max_steps)
    eos_id = _token_id(eos_id)
    if min(beam_width, max_steps) < 1:
        raise ValueError("beam width and generation cap must be positive")

    beam = [((), 0.0)]
    for _ in range(max_steps):
        candidates = []
        for prefix, log_probability in beam:
            if prefix and prefix[-1] == eos_id:
                candidates.append((prefix, log_probability))
                continue
            for token, p in _positive_probabilities(next_probabilities(prefix)):
                candidates.append((prefix + (token,), log_probability + math.log(p)))
        # Break equal-score ties by token ID order.
        beam = sorted(candidates, key=lambda item: (-item[1], item[0]))[:beam_width]
        if all(tokens[-1] == eos_id for tokens, _ in beam):
            break
    tokens, log_probability = beam[0]
    return tokens, log_probability, tokens[-1] == eos_id
