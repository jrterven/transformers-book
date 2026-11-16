# Chapter 32 | AI Agents with Language Models

The campus desk experiment accompanies Chapter 32. The workshop records are fictional. The native calls, returned tool values and terminal answers in the supplied JSON files come from the recorded runs of a local gpt-oss-20b model.

## Read the recorded experiment

Open [`MEASURED_RUN.json`](records/MEASURED_RUN.json) and compare `camera_updated` with `camera_cancelled`. For each step, `trace` gives the proposed tool, the arguments and the observation, or the terminal answer. `public_messages` gives the corresponding conversation and tool messages. The report includes the model digest, server version and generation settings.

The second tool returns a different notice in the two camera cases. With an active notice the model checks the new room and uses the calculator. With a cancellation it returns a final answer after the notice.

- [`CLARIFICATION_RUN.json`](records/CLARIFICATION_RUN.json) records a repeated Sketch Club case with a more explicit instruction about evidence.
- [`BUDGET_RUN.json`](records/BUDGET_RUN.json) records the active camera case with two permitted calls. The runner records the third proposal at the limit before dispatching it.
- [`FIXED_WORKFLOW.json`](records/FIXED_WORKFLOW.json) records the conditional baseline. This program receives the already extracted workshop title and group size for each case.

## Run the baseline

Use Python 3.10 or newer. The two runners use the standard library. From the companion repository root, run

```sh
python3 -m examples.ch32_agents.fixed_workflow --output new-baseline.json
```

Compare `checks`, `success` and `executed_calls` for each case with the saved runs.

## Run the language model

Start [Ollama](https://ollama.com/) and install `gpt-oss:20b` with `ollama pull gpt-oss:20b` if you need a local copy. The recorded artifact is approximately 13 GB. The runner checks the installed artifact against the model digest in the saved report. The recorded Ollama version was 0.40.16. Backend, artifact and prompt differences can affect the observed decisions.

```sh
python3 -m examples.ch32_agents.run_agent --output new-run.json
python3 -m examples.ch32_agents.run_agent --include sketch_ambiguous --clarification-evidence --output new-clarification.json
python3 -m examples.ch32_agents.run_agent --include camera_updated --max-calls 2 --output new-budget.json
```

The runner contacts the local Ollama API at `127.0.0.1:11434`. Python executes the four desk functions in `campus.py`. These functions read the supplied records or multiply and add integer cents. The runner writes one report for each command.

## Pause and resume the long-horizon example

[`HORIZON_RUN.json`](records/HORIZON_RUN.json) includes the saved waiting state and both measured continuations. In the example the notice becomes available at simulated minute 180 and the run expires at minute 240. Each command is a separate process. The saved public messages are sufficient to reconstruct the context for the next call. Use a fresh directory and run the following commands in order with the same local model.

```sh
mkdir horizon-demo
python3 -m examples.ch32_agents.resume_agent --checkpoint horizon-demo/waiting.json --minute 0
python3 -m examples.ch32_agents.resume_agent --checkpoint horizon-demo/waiting.json --minute 120
cp horizon-demo/waiting.json horizon-demo/original.json
cp horizon-demo/waiting.json horizon-demo/guarded.json
cp horizon-demo/waiting.json horizon-demo/expired.json
python3 -m examples.ch32_agents.resume_agent --checkpoint horizon-demo/original.json --minute 180
python3 -m examples.ch32_agents.resume_agent --checkpoint horizon-demo/guarded.json --minute 180 --guard
python3 -m examples.ch32_agents.resume_agent --checkpoint horizon-demo/guarded.json --minute 185
python3 -m examples.ch32_agents.resume_agent --checkpoint horizon-demo/expired.json --minute 240
```

Both recorded continuations start from the same two tool results. Inspect `trace` to see which room was proposed and whether it was actually checked. The execution rule adds a `proposal_rejected` event when a dependent call needs newer evidence. Rejected proposals consume an attempt. Invocations before the scheduled release or after completion leave the checkpoint bytes unchanged. The file example demonstrates sequential restarts on one machine.

## Redraw the figures

Install Matplotlib in your Python environment, then run

```sh
python3 -m examples.ch32_agents.make_figures --run examples/ch32_agents/records/MEASURED_RUN.json \
  --horizon-run examples/ch32_agents/records/HORIZON_RUN.json \
  --output output/ch32-figures
```

The command uses the recorded observations and token counts for the measured figures. It also redraws the explanatory diagrams as vector PDFs.

## Practice

Work through the [13 exercises](EXERCISES.md) and check the [solutions](../../solutions/ch32_agents/).
