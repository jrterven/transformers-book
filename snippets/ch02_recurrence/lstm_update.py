def gated_cell_update(cell, forget, write, candidate, expose):
    """The elementwise memory update after the gates have been calculated."""
    new_cell = forget * cell + write * candidate
    new_hidden = expose * np.tanh(new_cell)
    return new_hidden, new_cell
