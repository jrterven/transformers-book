# Chapter 30 exercises

Decisions and Embodied Action

Exercises 1-4 trace representations by hand; 5-8 use Python; 9-11 design evaluations.
All have [complete solutions](../../solutions/ch30_decisions/README.md).
Import helpers from `examples.ch30_decisions.model`.

The grid is 2 by 3, with zero-based row and column indices.
Start=(0,0), goal=(0,2), original trap=(0,1), shifted trap=(1,1).
The original log has three `TRAP_TRAJECTORY` copies and one `GOAL_TRAJECTORY`.
Goal route: D,R,R,U. Failed route: R. See `run_examples.py`.
The runnable command referenced as Listing 30.4 is:

```sh
python -m examples.ch30_decisions.run_examples --output output/ch30.json
```

1. **Compute returns-to-go.** A trajectory receives rewards $(0,2,-1,4)$. Write its return-to-go at each of the four steps. Check the recurrence after the negative reward.

2. **Identify the available tokens.** Write the tokens for steps 0, 1, and 2. With a context window that includes only the current step and the preceding step, mark which tokens the output at $s_2$ can read when predicting $a_2$. Explain where the initial return token comes from at training time and at deployment. Explain why $a_1$ is allowed but $a_2$ is blocked when predicting $a_2$.

3. **Count the two policies.** Use Figure 30.3 and the four trajectories described in the chapter. Count the logged actions at the start for the state-only fit and for the return-conditioned fit with target $+1$. Then change the desired return from $+1$ to $-1$ and state which part of the log can supply a matching action.

4. **Read an action bin.** A controller uses the ordered displacements $(-2,-1,0,1,2)$ cm with indices 0 through 4. Decode bins 1 and 4. Explain what would go wrong if the model and controller disagreed on the order of the bins.

5. **Check the return routine.** Use `returns_to_go` on the rewards in Exercise 1 and on a single-step trajectory with reward $-3$. Compare the first return with the sum of the rewards in each case. Check the empty input too. As an extension, compute discounted returns for Exercise 1 with $\gamma=1/2$ and verify their recurrence; do not change the undiscounted function.

6. **Test an unsupported target.** Import `TRAJECTORIES` from the runnable module in Listing 30.4. Pass it to `fit_count_policies`, then call `rollout` with the second table and `target_return=2`. Inspect the stop reason, visited states, and actions. Explain how these results differ from reaching the trap. Repeat with target 0. Explain why a missing entry does not by itself prove that a return is impossible.

7. **Inspect the shifted rollout.** Run Listing 30.4. Under `grid`, compare the conditioned rollout on the original grid with the one on the shifted grid. The state lists start with the initial position. Find the first action step whose resulting state or reward differs. Explain why the policy's first two actions agree.

8. **Vary the demonstrations.** Use `TRAP_TRAJECTORY` once and `GOAL_TRAJECTORY` three times to build a new log. Fit both count tables on this log. Run both on the original grid, with target $+1$ for the conditioned table. Compare what the state-only policy learns with the original result. Repeat with two copies of each trajectory and explain the tie-breaking result.

9. **Split entire episodes.** You have 100 trajectories collected across five grid layouts. Design separate tests for new starts on familiar layouts and for a layout excluded from training. Explain why splitting individual steps can leak information about an episode.

10. **Compare imitation with closed-loop success.** Two policies predict the same fraction of actions correctly on held-out demonstrations. Design a simulator test that could distinguish them when one mistake sends the agent to an unfamiliar position. Specify the starting conditions and outcome measures.

11. **Evaluate the control interface.** A robot receives a camera image and executes up to eight predicted actions before refreshing its observation. Use the same trained model and design a comparison that executes fewer actions from each predicted block before refreshing the image. Keep $H=8$ and compare $K=8$ with $K=2$. For actions lasting 0.1 s and ignoring inference latency temporarily, calculate the refresh intervals. Then specify measures of success, timing, and recovery under the same starting conditions, including the effect of real inference latency.
