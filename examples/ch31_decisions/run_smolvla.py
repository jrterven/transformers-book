"""Run the released SmolVLA policy on one LIBERO pick-and-place task."""

import argparse
from hashlib import file_digest
from importlib.metadata import version
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np


REPO = "lerobot/smolvla_libero"
REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"
WEIGHT_SHA256 = "9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8"
NORMALIZER_SHA256 = "b0cdde6e8a6f49a8e19eefb376728e47c09d3b3cc20ce3a97c45619fe7a732d9"
POLICY_FILES = {
    "model.safetensors": WEIGHT_SHA256,
    "config.json": "3ed3ed80ee077d94ff22942130642bb60e255443ee65da676bc88ab1684a7efe",
    "policy_preprocessor.json": "7beef33479fc22c9a45670154b3c7b909fb6d51dac3cd6db6f23376638b4ac11",
    "policy_postprocessor.json": "3b51f092c70c710ce0213ee1b63bf51b4878ec67828dd2d67daf9ef51081a41a",
    "policy_preprocessor_step_5_normalizer_processor.safetensors": NORMALIZER_SHA256,
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors": NORMALIZER_SHA256,
}
VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_REVISION = "4cde9d208e19519c36917aa170fffc7e381aa0ad"
ASSETS_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
SUITE = "libero_spatial"
TASK_ID = 0
TRIALS = (0, 1, 2)
HORIZONS = (5, 25)
CONTROL_HZ = 20
STEP_LIMIT = 280
FRAME_STEPS = (0, 46, 64, 76)


def model_inputs(obs, instruction):
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep

    robot = obs["robot_state"]
    batched = {
        "pixels": obs["pixels"],
        "robot_state": {
            "eef": {key: robot["eef"][key][None, :] for key in ("pos", "quat")},
            "gripper": {"qpos": robot["gripper"]["qpos"][None, :]},
        },
    }
    batch = LiberoProcessorStep().observation(preprocess_observation(batched))
    batch["task"] = [instruction]
    return batch


def read_geometry(env, obs):
    data = env._env.sim.data
    return (
        obs["robot_state"]["eef"]["pos"].tolist(),
        obs["robot_state"]["gripper"]["qpos"].tolist(),
        data.get_body_xpos("akita_black_bowl_1_main").tolist(),
        data.get_body_xpos("plate_1_main").tolist(),
    )


def run_episode(policy, preprocess, postprocess, initial_state, horizon, frame_dir):
    import torch
    from lerobot.envs.libero import LiberoEnv
    from libero.libero import benchmark
    from PIL import Image

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    env = LiberoEnv(
        task_suite=benchmark.get_benchmark_dict()[SUITE](),
        task_suite_name=SUITE, task_id=TASK_ID, episode_index=initial_state,
        obs_type="pixels_agent_pos", control_mode="relative", control_freq=CONTROL_HZ,
    )
    actions, rewards, trace, queries = [], [], [], []
    clipped = 0
    success = terminated = False
    torch.manual_seed(3000 + initial_state)
    policy.reset()
    start = perf_counter()
    if frame_dir is not None:
        frame_dir.mkdir(parents=True, exist_ok=True)

    def record(obs):
        eef, fingers, bowl, plate = read_geometry(env, obs)
        trace.append({"eef": eef, "fingers": fingers, "bowl": bowl, "plate": plate})
        step = len(trace) - 1
        if frame_dir is not None and step in FRAME_STEPS:
            for camera, pixels in obs["pixels"].items():
                Image.fromarray(pixels[::-1, ::-1]).save(frame_dir / f"{step:03d}-{camera}.png")

    try:
        obs, _ = env.reset(seed=300 + initial_state)
        controller = env._env.robots[0].controller
        limits = {
            "input_min": controller.input_min.tolist(), "input_max": controller.input_max.tolist(),
            "output_min": controller.output_min.tolist(), "output_max": controller.output_max.tolist(),
        }
        record(obs)
        while len(actions) < STEP_LIMIT and not terminated:
            query_start = perf_counter()
            batch = preprocess(model_inputs(obs, env.task_description))
            normalized = policy.predict_action_chunk(batch)
            chunk = postprocess(normalized[0]).numpy()
            if chunk.shape != (policy.config.chunk_size, 7) or not np.isfinite(chunk).all():
                raise ValueError("expected one finite seven-component command per predicted step")
            elapsed = perf_counter() - query_start
            queries.append({
                "step": len(actions), "seconds": elapsed,
                "first_normalized": normalized[0, 0].tolist(), "first_command": chunk[0].tolist(),
            })
            remaining = min(horizon, STEP_LIMIT - len(actions))
            for command in chunk[:remaining]:
                action = np.clip(command, env.action_space.low, env.action_space.high)
                clipped += int(np.any(action != command))
                obs, reward, ended, truncated, info = env.step(action)
                actions.append(action.tolist())
                rewards.append(float(reward))
                record(obs)
                success = bool(info["is_success"])
                terminated = bool(ended or truncated)
                if terminated:
                    break
        return {
            "instruction": env.task_description, "initial_state": initial_state, "horizon": horizon,
            "success": success, "terminated": terminated, "step_limit": not terminated,
            "control_hz": CONTROL_HZ, "actions": actions, "rewards": rewards, "trace": trace,
            "queries": queries, "clipped": clipped, "controller": limits,
            "wall_seconds": perf_counter() - start,
        }
    finally:
        env.close()


def load_policy(checkpoint, vlm):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    files = (
        *((checkpoint / name, sha) for name, sha in POLICY_FILES.items()),
        (vlm / "config.json", "ea6bc1237e96247f6258de3e202e2e62b93d6f386dc47e7b36b5588bf3a15e17"),
        (vlm / "tokenizer.json", "5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a"),
    )
    for path, expected in files:
        with path.open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"SmolVLA file differs from the pinned revision: {path.name}")
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = "cpu"
    # The complete policy checkpoint includes its VLM tensors.
    config.load_vlm_weights = False
    config.vlm_model_name = str(vlm)
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, local_files_only=True, strict=True)
    preprocess, postprocess = make_pre_post_processors(
        config, str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "tokenizer_processor": {"tokenizer_name": str(vlm)},
        },
    )
    return policy, preprocess, postprocess


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--initial-states", type=int, nargs="+", choices=TRIALS, default=TRIALS)
    parser.add_argument("--horizons", type=int, nargs="+", choices=HORIZONS, default=HORIZONS)
    args = parser.parse_args()
    torch.set_num_threads(6)
    policy, preprocess, postprocess = load_policy(args.checkpoint, args.vlm_config)
    episodes = []
    for initial_state in args.initial_states:
        for horizon in args.horizons:
            frame_dir = None
            if args.frames is not None and (initial_state, horizon) == (TRIALS[0], HORIZONS[0]):
                frame_dir = args.frames
            row = run_episode(policy, preprocess, postprocess, initial_state, horizon, frame_dir)
            episodes.append(row)
            print(initial_state, horizon, row["success"], len(row["actions"]), len(row["queries"]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": REPO, "revision": REVISION, "weights_sha256": WEIGHT_SHA256,
        "vlm_repo": VLM_REPO, "vlm_revision": VLM_REVISION, "assets_revision": ASSETS_REVISION,
        "device": "cpu", "threads": 6, "suite": SUITE, "task_id": TASK_ID,
        "step_limit": STEP_LIMIT, "predicted_horizon": policy.config.chunk_size,
        "flow_steps": policy.config.num_steps, "control_hz": CONTROL_HZ,
        "versions": {name: version(name) for name in
                     ("torch", "numpy", "lerobot", "transformers", "mujoco", "robosuite", "hf-libero")},
        "episodes": episodes,
    }, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
