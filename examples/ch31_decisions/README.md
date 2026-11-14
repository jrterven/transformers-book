# Chapter 31. Decisions and Embodied Action

Run the fixed examples from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch31_decisions.run_examples --output output/ch31.json
python -m unittest discover -s tests -p test_decisions.py -v
```

`returns_to_go` adds each reward to all the rewards that follow
it in the same trajectory. The rewards `[1, -2, 3]` give the
targets `[2, 1, 3]`. An empty sequence gives an empty result.

The four fixed training trajectories in `run_examples.py` record
`(state, action, reward)` at each step. Grid cells use zero-based
`(row, column)` coordinates. The start is `(0, 0)`, the goal is
`(0, 2)`, and the original trap is `(0, 1)`. Reaching the goal
gives `+1`, reaching the trap gives `-1`, and all other moves give
zero. A move into a wall leaves the agent in place.

`fit_count_policies` returns two dictionaries. The first chooses
the most frequent action observed at a state. The second chooses
the most frequent action observed at a state with a given return
to go. Ties use `U, R, D, L` in that order. The conditioned table
looks up an exact target value, which suits the grid's integer
rewards.

`rollout` takes one of the fitted dictionaries and asks it for an
action at each newly observed state. For the conditioned table,
pass `target_return`. After each move, the trace subtracts its
reward from the remaining target. The trace includes the initial
state and, when supplied, the initial target. A missing table entry
stops the rollout with `unsupported` before another action.

Three of the recorded trajectories take `R` into the trap. The
state-only table follows this majority. At target `+1`, the
conditioned table follows the goal route `D, R, R, U` on the
original grid. The same table follows `D, R` into the trap when it
moves to `(1, 1)`. Both tables are simple counts over the recorded
actions. They illustrate how a desired return changes the lookup
and how a changed environment can change the outcome.

## What the examples establish

The reward index refers to the action that earns it. Recorded returns describe
completed episodes; requested returns condition a policy. Neither a large request
nor a valid action token guarantees a successful outcome.

The shifted trap is omitted from the policy inputs. Equal position/return keys
therefore produce equal first actions, despite different consequences.
An unsupported target is distinct from a terminal trap or a time limit.
Target 0 is absent from the log but attainable by a wall policy at the step cap;
target +2 is impossible under this grid's terminal rewards.

The notebook adds a discounted-return extension without changing the canonical
undiscounted helper. It also checks tied demonstrations and separates an
eight-action prediction horizon from the number executed before refreshing.

See the [11 exercises](EXERCISES.md) and
[complete solutions](../../solutions/ch31_decisions/README.md).

## Run the pretrained policies

Vintix II chooses two motor commands for a Reacher-v4 arm. For reset seeds
30–32, we run the same checkpoint once with every completed transition
available and once with the current observation alone. Both runs observe
the simulator after each command. The initial observation and action agree
within each pair. The return is higher with history on two of the three
resets. Reacher-v4 charges the tip distance *before* the action and the
sum of the two squared commands. It uses a 50-action time limit, which is
1 simulated second.

SmolVLA reads two cameras, the language instruction and eight robot
measurements. It generates 50 seven-component commands with 10 flow steps.
The rollout applies the checkpoint's action postprocessor and clips the
result to the simulator's bounds. We compare using 5 or 25 commands before
the next prediction, with the same fixed starts. The simulator marked 3/3
and 2/3 episodes successful respectively. These six trials use one task
on which the checkpoint was fine-tuned. The episode ends at the first
positive goal predicate or after 280 actions.

### Redraw the figures from the saved runs

[Vintix II trace](records/vintix_reacher.json) and
[SmolVLA trace](records/smolvla_libero.json) contain the actual actions,
rewards, and simulator coordinates. The SmolVLA trace also records
inference durations. They are local CPU measurements and include
observation preprocessing, inference, and action decoding. The
[frames](records/frames/) are rendered observations from the successful
initial-state-0 rollout at an execution horizon of 5. They are attributed to
the [LIBERO benchmark](https://github.com/huggingface/LIBERO) and
[LeRobot's LIBERO assets](https://huggingface.co/datasets/lerobot/libero-assets).

```sh
python -m pip install "numpy==2.2.6" "matplotlib==3.11.2"
python -m examples.ch31_decisions.make_figures --output-dir output/ch31-figures
```

### Reproduce the Vintix II rollout

Run commands from the companion repository root in a Python 3.12 virtual
environment. The first demo uses the released
[Vintix II model](https://huggingface.co/dunnolab/VintixII/tree/fd9181953893806a3065b89ca7c80651a2d08351)
and [source revision](https://github.com/dunnolab/vintix-II/tree/5e185d9c660f6f4e89c4d1a3483bd4e79332530f).
The 3.7 GB download contains the original weights. For this pinned source,
the one-line upstream import repair below points to the included module.
The runner uses PyTorch's causal scaled dot-product attention and samples
the last output with the original flow head on CPU.

```sh
python -m pip install "torch==2.11.0" "numpy==2.2.6" \
  "gymnasium[mujoco]==1.4.0" "mujoco==3.3.2" "imageio==2.38.1" \
  "huggingface-hub==1.33.0"
git clone https://github.com/dunnolab/vintix-II.git output/vintix-II
git -C output/vintix-II checkout 5e185d9c660f6f4e89c4d1a3483bd4e79332530f
python - <<'PY'
from pathlib import Path
p = Path("output/vintix-II/vintix/model/model.py")
source = p.read_text()
old = "from src.model.nn import"
assert source.count(old) == 1
p.write_text(source.replace(old, "from vintix.model.nn import"))
PY
hf download dunnolab/VintixII config.json metadata.json model.pth \
  --revision fd9181953893806a3065b89ca7c80651a2d08351 --local-dir output/vintix-weights
PYTHONPATH=output/vintix-II python -m examples.ch31_decisions.run_vintix \
  --checkpoint output/vintix-weights --output output/vintix-replay.json
```

The seed fixes the Gaussian draw at each corresponding action in the two
runs. The history condition keeps the observations, executed actions and
realized rewards. It uses the task's original input encoders and weights.

### Reproduce the SmolVLA rollout

Use a fresh Python 3.12 environment. On macOS the rendering session needs
access to the logged-in display. Install the following subset of LIBERO and
robosuite dependencies for the simulator and inference path. For a Linux
renderer, use the [LeRobot LIBERO setup](https://huggingface.co/docs/lerobot/libero)
with EGL or OSMesa. Keep MuJoCo at 3.3.2 for the version used here.

```sh
python -m pip install -r examples/ch31_decisions/requirements-pretrained.txt
python -m pip install --no-deps "hf-libero==0.1.4" "robosuite==1.4.0" "bddl==1.0.1"
python -m pip install "numba==0.68.1" "llvmlite==0.50.0" "scipy==1.18.1" \
  "easydict==1.14" "future==1.0.0" "h5py==3.16.0"
hf download lerobot/smolvla_libero --revision a24f1f380461b665158d52acd140152c79e8b36d \
  --local-dir output/smolvla
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --revision 4cde9d208e19519c36917aa170fffc7e381aa0ad \
  --exclude "*.safetensors" "onnx/*" --local-dir output/smolvlm-config
hf download lerobot/libero-assets --repo-type dataset \
  --revision 0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 --local-dir output/libero-assets
```

Set up a private LIBERO configuration for this run. The path printed by
the script belongs to the active virtual environment.

```sh
python - <<'PY'
from importlib.util import find_spec
from pathlib import Path
import yaml

root = Path(find_spec("libero").submodule_search_locations[0]) / "libero"
assets = Path("output/libero-assets").resolve()
if not (root / "assets").exists():
    (root / "assets").symlink_to(assets, target_is_directory=True)
config = Path("output/ch31-libero-config").resolve()
config.mkdir(parents=True, exist_ok=True)
(config / "config.yaml").write_text(yaml.safe_dump({
    "benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
    "init_states": str(root / "init_files"), "assets": str(assets),
    "datasets": str(Path("output/libero-data").resolve()),
}))
PY
HF_HUB_OFFLINE=1 LIBERO_CONFIG_PATH="$PWD/output/ch31-libero-config" \
  python -m examples.ch31_decisions.run_smolvla \
  --checkpoint output/smolvla --vlm-config output/smolvlm-config \
  --output output/smolvla-replay.json --frames output/smolvla-frames
```

The 907 MB policy checkpoint contains its own vision-language weights.
The SmolVLM download supplies its pinned tokenizer and architecture
configuration. The runner checks the weight and processor hashes and
loads the policy strictly. The postprocessor maps standardized values
to controller units using the checkpoint's recorded mean and standard
deviation. The two schedules start with the same flow seed per reset.
Wall-clock timings vary with the machine and renderer.
