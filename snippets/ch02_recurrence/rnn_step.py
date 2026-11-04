def rnn_trace(inputs, recurrent_weight=0.5, initial_state=0.0):
    """Reuse one scalar update at every position, returning every state."""
    state = float(initial_state)
    states = []
    for value in inputs:
        state = np.tanh(recurrent_weight * state + float(value))
        states.append(state)
    return np.asarray(states)
