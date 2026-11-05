def hyena_stage(u, input_gate, output_gate, kernel):
    """Return the gated input, filtered sequence, and gated output."""
    u = _real_array(u, "u", 1)
    input_gate = _real_array(input_gate, "input gate", 1)
    output_gate = _real_array(output_gate, "output gate", 1)
    if not (u.shape == input_gate.shape == output_gate.shape):
        raise ValueError("each gate needs one entry per input position")

    with np.errstate(over="raise", invalid="raise"):
        gated_input = input_gate * u
        filtered = causal_convolution(gated_input, kernel)
        output = output_gate * filtered
    return gated_input, filtered, output
