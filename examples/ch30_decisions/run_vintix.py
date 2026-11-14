"""Run the released Vintix II weights in the two-joint Reacher simulator."""

import argparse
from hashlib import file_digest
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np


REPO = "dunnolab/VintixII"
REVISION = "fd9181953893806a3065b89ca7c80651a2d08351"
SOURCE_REVISION = "5e185d9c660f6f4e89c4d1a3483bd4e79332530f"
WEIGHT_SHA256 = "0343f7bacd96f9087bfe66a47234665e475df392a253954a4e8e1d31c2acdeda"
CHECKPOINT_HASHES = {
    "model.pth": WEIGHT_SHA256,
    "config.json": "0419e94a6bea0a31d374e6ab377088e357477cdc484fefbde7e72aa8c76289b2",
    "metadata.json": "3b392333f7143e4050e2dd909d47983eb32e4f35450e60336da9d6c332a97341",
}
TASK = "Reacher-v4"
SEEDS = (30, 31, 32)


def action_from_context(model, observations, actions, rewards):
    """Read the last Transformer output and sample one action at that position."""
    import torch

    # The upstream encoder expects a dummy previous action and reward first.
    inputs = [{
        "observation": torch.tensor(np.asarray(observations), dtype=torch.float32),
        "prev_action": torch.tensor(np.asarray([[0.0, 0.0], *actions]), dtype=torch.float32),
        "prev_reward": torch.tensor([0.0, *rewards], dtype=torch.float32).unsqueeze(1),
        "perm": np.arange(len(actions)),
        "task_name": TASK,
    }]
    with torch.inference_mode():
        encoded = model.encoder(inputs)
        hidden = model.transformer(encoded["input_tokens"])
        # The flow head operates independently at each sequence position.
        # Keep its expected BOS placeholder and sample the final position only.
        query = hidden[:, -1:, :]
        if model.head.add_bos:
            query = torch.cat((hidden[:, :1, :], query), dim=1)
        return model.head(query, encoded["metadata"])[0][0].numpy()


def rollout(model, seed, use_history):
    import gymnasium as gym
    import torch

    env = gym.make(TASK)
    observation, _ = env.reset(seed=seed)
    observations, commands, rewards = [observation], [], []
    targets = observation[4:6].copy()
    tip = [env.unwrapped.get_body_com("fingertip")[:2].copy()]
    clipped = 0
    generator = torch.Generator().manual_seed(10_000 + seed)
    action_seeds = torch.randint(0, 2**31, (env.spec.max_episode_steps,), generator=generator)
    try:
        for action_seed in action_seeds:
            torch.manual_seed(action_seed.item())
            if use_history:
                proposed = action_from_context(model, observations, commands, rewards)
            else:
                proposed = action_from_context(model, observations[-1:], [], [])
            if proposed.shape != (2,) or not np.isfinite(proposed).all():
                raise ValueError("the model must return two finite motor commands")
            action = np.clip(proposed, env.action_space.low, env.action_space.high)
            clipped += int(np.any(action != proposed))
            observation, reward, terminated, truncated, info = env.step(action)
            expected_reward = -np.linalg.norm(observations[-1][-3:]) - np.dot(action, action)
            if not np.isclose(reward, expected_reward, atol=1e-6):
                raise ValueError(f"unexpected Reacher reward convention: {info}")
            observations.append(observation)
            commands.append(action.tolist())
            rewards.append(float(reward))
            tip.append(env.unwrapped.get_body_com("fingertip")[:2].copy())
            if terminated or truncated:
                break
    finally:
        env.close()
    return {
        "seed": seed, "history": use_history, "target": targets.tolist(),
        "tip_xy": np.asarray(tip).tolist(), "actions": commands, "rewards": rewards,
        "clipped": clipped, "terminated": bool(terminated), "truncated": bool(truncated),
        "return": sum(rewards), "period_s": float(env.unwrapped.dt),
    }


def main():
    import torch
    from vintix import Vintix2
    from .vintix_cpu import use_cpu_attention

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name, expected in CHECKPOINT_HASHES.items():
        with (args.checkpoint / name).open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Vintix II file differs from the pinned revision: {name}")
    torch.set_num_threads(6)
    use_cpu_attention()
    model = Vintix2()
    model.load_model(str(args.checkpoint))
    model.eval()
    episodes = []
    for seed in SEEDS:
        for history in (True, False):
            row = rollout(model, seed, history)
            episodes.append(row)
            print(seed, history, len(row["actions"]), round(row["return"], 3), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": REPO, "revision": REVISION, "weights_sha256": WEIGHT_SHA256,
        "source_revision": SOURCE_REVISION, "environment": TASK,
        "versions": {name: version(name) for name in ("torch", "numpy", "gymnasium", "mujoco")},
        "device": "cpu", "threads": 6, "flow_steps": model.head.steps, "episodes": episodes,
    }, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
