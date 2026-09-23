# RL vs. Distilled Reasoners

Do the reasoning mechanisms found in distilled reasoning models also hold in
models that learned to reason through RL?

[Read the blog here.](https://www.divinrkz.com/blog/research/rl-vs-distillation-reasoners)

Most interpretability results on reasoning models (Thought Anchors, reasoning
steering vectors) were measured on R1 distills. This repo checks two of them
across training regimes, using a matched 7B pair:

- **RL:** `Open-Reasoner-Zero/Open-Reasoner-Zero-7B`
- **Distilled:** `open-thoughts/OpenThinker-7B`

## Experiments

1. **Thought Anchors.** Resample from each sentence of a CoT, measure its
  counterfactual importance, and aggregate by sentence category (Bogdan et al.,
   2025).
2. **Steering vectors.** Extract a backtracking vector from OpenThinker, verify
  it steers OpenThinker, then test transfer to ORZ against a random
   matched-norm control (Venhoff et al., 2025).


## Findings

- **Exp 1:** ORZ's important sentences are setup and knowledge recall; its
backtracking and uncertainty importance is zero. OpenThinker's important
sentences are deduction, with nonzero backtracking and uncertainty.
- **Exp 2:** The OpenThinker vector steers OpenThinker coherently but does not
transfer to ORZ at coherent strengths. The random control shows no effect.

Results are underpowered by design (~7–10 problems, 10 rollouts per sentence).
Divergence is informative; a null would not be. See the blog post for caveats.

## Layout

```
src/rl_distill/   sentence splitting, grading, importance, labeling, steering
scripts/          numbered pipeline steps (01_smoke … 06_plots)
configs/          one YAML per run
experiments/      outputs + config copy per run (gitignored)
figures/          Figure 1 (category profiles), Figure 2 (steering curves)
notes/            running log and hypotheses
```

