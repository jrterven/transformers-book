def fit_count_policies(trajectories):
    """Count actions from trajectories of (state, action, reward) rows."""
    by_state = defaultdict(Counter)
    by_return = defaultdict(Counter)
    for trajectory in trajectories:
        trajectory = tuple(trajectory)
        targets = returns_to_go([reward for _, _, reward in trajectory])
        for (state, action, _), target in zip(trajectory, targets):
            state = _grid_cell(state)
            if not isinstance(action, str) or action not in MOVES:
                raise ValueError("unknown action")
            by_state[state][action] += 1
            by_return[state, float(target)][action] += 1

    state_policy = {key: max(ACTIONS, key=counts.__getitem__)
                    for key, counts in by_state.items()}
    return_policy = {key: max(ACTIONS, key=counts.__getitem__)
                     for key, counts in by_return.items()}
    return state_policy, return_policy
