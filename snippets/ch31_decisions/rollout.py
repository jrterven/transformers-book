state = START
states, actions, rewards = [state], [], []
remaining_targets = None if target_return is None else [target_return]
stop_reason = "step_limit"
for _ in range(max_steps):
    key = state if target_return is None else (state, target_return)
    action = policy.get(key)
    if action is None:
        stop_reason = "unsupported"
        break
    if not isinstance(action, str) or action not in MOVES:
        raise ValueError("unknown action")

    state, reward = _grid_step(state, action, trap)
    states.append(state)
    actions.append(action)
    rewards.append(reward)
    if remaining_targets is not None:
        target_return -= reward
        remaining_targets.append(target_return)
    if state in (GOAL, trap):
        stop_reason = "goal" if state == GOAL else "trap"
        break
