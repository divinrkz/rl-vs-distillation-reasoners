# Running log

## Step 1 — smoke test + format diff

Scaffolded `src/rl_distill/` (formats, generate, grading, parsing, runio) and `scripts/01_smoke.py`. Formats are declarative in `configs/smoke.yaml`.

**To run on the GPU box:**

```
pip install -r requirements.txt
python scripts/01_smoke.py configs/smoke.yaml
```

Writes `experiments/smoke/{orz,openthinker}/results.json` + a config copy.

**Must verify by eye before Step 2 (formats are the #1 silent-failure risk):**

- ORZ prompt template and whether it actually emits `<think>`/`<answer>` tags. The default in `smoke.yaml` seeds `Assistant: <think>` — confirm against the ORZ model card. If ORZ uses a different template, edit `formats.orz`.
- OpenThinker: does it emit `<think>`/`</think>`, or the OpenThoughts `<|begin_of_thought|>` markers? If the latter, edit `formats.openthinker` think tags. Confirm whether its chat template already injects a reasoning system prompt (then `system_prompt: null` is correct).
- `extract_thinking` / `extract_answer` output on both models looks sane.



### Step 1 eye-diff findings (first smoke run)

Two silent bugs caught, both fixed:

**Grader false negatives.** Both models answered the polar-coordinates problem correctly (`(3, \pi/2)`) but were marked wrong. Cause: math-verify was given bare, undelimited LaTeX and failed to verify the tuple. Fixed in  
 `grading.py` (`_as_latex` wraps bare answers; grade from the completion so math-verify extracts `\boxed{}` itself). Verify with`python scripts/check_grader.py` (CPU-only).

1. **OpenThinker tag mismatch.** OpenThinker uses
  `<|begin_of_thought|>/<|end_of_thought|>` + `<|begin_of_solution|>/  <|end_of_solution|>`, NOT `<think>`. The wrong tags made `extract_thinking` return the whole completion (solution + boxed answer included). Fixed  `formats.openthinker` in `smoke.yaml`. ORZ (`<think>`/`<answer>`) parsed correctly and is unchanged.

Behavioral note for the write-up: ORZ's CoT is short, formal, linear; OpenThinker's is longer and chattier with explicit self-checks ("Wait, let me double-check", "just to be thorough"). Category-profile difference looks real even at n=1 — the thing Exp 1 is meant to quantify.

## Step 2 — sentence splitting, both formats

`src/rl_distill/sentences.py` adapts thought-anchors `string_to_sentences` / `process_text_segment` / `clean_python_string_literal` (pkld cache + paragraph / token-range helpers dropped). `split_cot(text, fmt)` isolates the reasoning region via the format adapter first, so the splitter needs no per-format logic — both `<think>` (ORZ) and `<|begin_of_thought|>` (OpenThinker) work through the same path. Local test confirms no answer/solution leakage into sentences.

**Verify on the box (CPU):** `python scripts/02_split_check.py configs/smoke.yaml`
Eyeball: sentence boundaries sensible, no `\boxed`/solution text in either
model's sentences, math expressions not shredded mid-formula.

Bug found + fixed: decimals were split mid-number (`0.125` -> `0.` + `125`,
`1.64493` -> `1.` + `64493`) because the splitter treats period-before-digit as
a boundary. Added decimal protection (`(?<=\d)\.(?=\d)` -> placeholder) in
`process_text_segment`, mirroring the abbreviation protection. Genuine
boundaries (`.`  + digit, with space) still split.

Not bugs: standalone display equations become one-line sentences (fine); OpenThinker
reasoning legitimately contains `\boxed{}` (final-answer reasoning inside the
thought section, not solution-section leakage).

Same-problem sentence counts (double-sum problem): ORZ 33 vs OpenThinker 119.
OpenThinker derives the answer early then spends ~75 sentences numerically
verifying. Real behavioral divergence; also why length normalization matters.

## Step 3 — full Exp 1 pipeline, one problem / one model

Decisions: labeler = Qwen2.5-32B-Instruct-AWQ (~20GB, fits the L40S);
taxonomy = Venhoff 6-category (shared with Exp 2 steering vectors).

New: `labeling.py` (per-CoT labeler call -> one category per sentence),
`importance.py` (resampling importance `acc[i+1]-acc[i]`, category profile),
`formats.build_resample_prompt` (format-correct continuation prompt),
`sentences.split_cot_with_offsets`, `scripts/03_exp1_one.py`, `configs/exp1.yaml`.

Metric = resampling importance (accuracy). Counterfactual semantic-dedup
refinement deferred to the full run (needs an embedding model). All pure-Python
logic unit-checked locally (prefix reconstruction exact for both formats).

**Two-phase run (GPU), one model per process:**

```
python scripts/03a_exp1_generate.py configs/exp1.yaml   # target -> generation.json
python scripts/03b_exp1_label.py    configs/exp1.yaml   # labeler -> result.json
```

Watch: (1) base_correct=True (else importance is on a wrong trace); (2) wall-clock
of the resample step — extrapolate to ~10 problems x both models; (3) labeler
returns valid JSON (labeler_raw + low "unknown" count); (4) importance not all zero
(would mean rollouts always/never correct — raise resample n or pick a
non-trivial problem).

Why two processes: vLLM reserves `gpu_memory_utilization` (~40GB) as a KV pool on
load and does NOT release it reliably mid-process; also `free_llm`'s `del` only
drops the local binding, not the caller's reference. Loading target then labeler
in one process OOM'd (4GB free). Splitting into two scripts frees the GPU on
process exit. (Smoke's sequential loads work only because each load lives in a
function scope that ends before the next.)

### Step 3 first result (ORZ, MATH-500 index 15, Level 5)

Problem selection matters: index 0 (easy) gave acc all 1.0, flat importance.
Added `probe_difficulty.py` to find intermediate pass-rate problems; index 15
(ORZ pass ~0.56) has signal. base_correct=True, 40 sentences.

acc[0]=0.5 -> importance[0]=0.5: sentence 0 is a dominant anchor (fixing the
opening move takes accuracy 0.5 -> 1.0). Everything after is +/-0.1 = the n=10
noise floor.

Category profile: initializing 0.385 (n=1, the real anchor), deduction 0.231
(n=24), adding-knowledge 0.385 (n=15). The deduction/adding-knowledge shares are
NOISE accumulated over many sentences, not real anchors. example-testing,
uncertainty-estimation, backtracking all n=0 — ORZ did none on this problem
(consistent with its linear style). Category *absence* is robust and is the
divergence-relevant observation to compare against OpenThinker.

TODO (Step 5): noise-aware importance aggregation — thresholding abs(importance)
near the 1/n floor (or subtracting expected noise) so category profiles reflect
real anchors, not sentence counts. Also consider raising resample n to 15-20 to
lower the noise floor.

## Step 4 — steering extraction + within-model control

Extract from OpenThinker first (ORZ has ~0 backtracking/uncertainty). HF
transformers + raw forward hooks (residual stream unreachable via vLLM).
Method: difference-of-means, vec[cat][layer] = mean(category tokens) -
mean(all reasoning tokens).

New: `steering/hooks.py` (capture + steer context managers on decoder layers),
`steering/extract.py` (char-span -> token-position mapping, diff-of-means),
`steering/apply.py` (generate with steering), `hf.py` (HF loader). Scripts
04a (traces, vLLM) -> 04b (label, vLLM) -> 04c (extract + steered/unsteered gen, HF).
Config `steer.yaml`: categories backtracking + uncertainty-estimation, layers
12-18, coeffs [0,4,8], eval indices 20-24.

**Run (GPU):**

```
python scripts/04a_gen_traces.py        configs/steer.yaml
python scripts/04b_label_traces.py      configs/steer.yaml
python scripts/04c_extract_and_steer.py configs/steer.yaml
```

Gates: (1) 04b backtracking + uncertainty token counts > 0 (need enough to
extract); (2) 04c vector norms non-zero; (3) eyeball control_outputs.json —
does coeff 4/8 make OpenThinker backtrack/hedge MORE than coeff 0, while staying
coherent? Too strong -> gibberish (lower coeff); no effect -> raise coeff or
change layers. 04d (quantified measurement via labeler) built after the eyeball
confirms a visible effect.

### Step 4 CONTROL PASSED (OpenThinker backtracking vector)

Extraction: backtracking 896 tokens, uncertainty 471, norms ~32/38 (healthy).
Steering at 7 layers compounds hard — coeff 8 collapsed to repeated Chinese
gibberish, coeff 1 already looped. Coherent band is FRACTIONAL. Dose-response on
backtracking (index 20, layers 12-18):
  coeff 0   : normal solve (some natural backtracking)
  coeff 0.2 : coherent, visibly MORE self-checking ("Wait, let me confirm")
  coeff 0.4 : backtracking dominates, coherence fraying
  coeff 0.6+: saturated doubt loops
Monotonic increase in backtracking language with coeff = the vector is causally
effective. Within-model control passes.

04d quantifies it: label steered outputs, measure steered-category sentence
fraction vs coeff (dose-response). Config now: both categories, coeffs [0,0.2,0.4],
eval indices 20-22.

Key number for later: OpenThinker's coherent steering window is narrow (~0.2-0.4
at 7 layers). Note when testing cross-model transfer to ORZ.

Quantified control (steered-category sentence fraction, mean over eval 20-22):
  backtracking:  coeff 0 -> 0.040, 0.2 -> 0.097, 0.4 -> 0.039
  uncertainty:   coeff 0 -> 0.043, 0.2 -> 0.087, 0.4 -> 0.211
Uncertainty is monotonic (clean). Backtracking doubles at 0.2 then drops at 0.4:
over-steered output degrades into doubt-loops phrased as uncertainty, which the
labeler tags as uncertainty (adjacent category) -> backtracking mass bleeds into
uncertainty. OPERATING POINT = coeff 0.2 (coherent, ~2x target). STEP 4 DONE.

### Cross-model transfer (OpenThinker vectors -> ORZ) — DIVERGENCE

ORZ backtracking fraction under OpenThinker's backtracking vector:
  coeff 0 -> 0.000, 0.2 -> 0.000, 0.4 -> 0.228, 0.8 -> 0.667
ORZ uncertainty under OpenThinker's uncertainty vector:
  coeff 0 -> 0.000, 0.2 -> 0.000, 0.4 -> 0.000, 0.8 -> 0.333

At coeff 0.2 (where the vector coherently steers OpenThinker, fraction 0.097),
ORZ shows ZERO response. ORZ only moves at 0.4/0.8 where output is fraying /
broken-grammar (eyeball). No coherent operating point for the transferred vector
=> mechanistic divergence: OpenThinker's reasoning steering vectors do not
transfer to the RL-trained model at a coherent strength.

Caveats: (1) coherence confound — 0.4/0.8 ORZ numbers are on degraded text;
labeler counts vocabulary not coherent reasoning. (3) Only OT->ORZ tested; ORZ->OT
extraction infeasible (ORZ has ~0 backtracking, itself a finding). (4) base-vs-
instruct + CoT-format confounds. (5) residual-norm scale may differ per model, so
a fixed coeff is a different relative strength — but ORZ has NO coherent band at
any coeff, so the qualitative claim survives.

Random matched-norm control (transfer_random -> ORZ): 0.000 at ALL coeffs, both
categories. => ORZ's 0.4/0.8 response is SPECIFIC to the backtracking direction,
not nonspecific degradation. EXP 2 COMPLETE.

Exp 2 result: within-model control works (OT vector steers OT, coherent);
OT->ORZ transfer fails at coherent strength (0 at coeff 0.2); effect is specific
(random=0) but only expressed incoherently at high coeff. => mechanistic
FIDELITY REJECTED; RL and distilled models encode backtracking in related but
non-aligned ways. Figure 2 = these three dose-response curves.

## Step 5 — Exp 1 at scale (Figure 1)

Built: `05_select.py` (intersect both models' probe pass rates -> both-solvable
intermediate problems), `05a_exp1_generate.py` (one model resident, loops
problems, resume-safe per-problem files), `05b_exp1_label.py` (batched labeling),
`05c_exp1_aggregate.py` (noise-thresholded per-category profile, mean+/-std across
problems -> profiles.json). `probe_difficulty.py` gained --model-id and --save.
`importance.py` gained threshold_importance + aggregate_profiles. `exp1_scale.yaml`.

Noise handling: threshold |imp| > noise_threshold (0.2 ~ 1/n at n=10) before
aggregating, so many-sentence categories don't accumulate noise (the Step 3
artifact). Per-problem profiles averaged equally (normalizes CoT length).

Run: probe both -> 05_select -> set indices -> 05a/05b per model -> 05c.
BUDGET: OpenThinker resampling is the expensive part (long CoTs, many prefixes).
Start with ~5 problems to gauge cost; 05a is resume-safe so indices can be
expanded incrementally.

### Figure 1 result (7 problems each, thresholded |imp|>0.2, mean+/-std)

```
          ORZ(RL)        OpenThinker(distilled)
```

initializing  0.262+/-0.370  0.067
deduction     0.059          0.427+/-0.386
adding-know   0.184          0.111
example-test  0.067          0.000
uncertainty   0.000          0.032
backtracking  0.000          0.078

Divergence: ORZ importance on initializing + adding-knowledge (setup/recall),
ZERO backtracking/uncertainty. OpenThinker importance on deduction, nonzero
backtracking + uncertainty. Converges with Exp 2 (OT backtracking vector fails
to transfer to ORZ). => mechanistic fidelity rejected.

Caveats: large std, columns sum <1 (with n=10 + threshold 0.2 many problems have
no suprathreshold anchor -> zero profile). Magnitudes underpowered at 7 problems;
the robust signal is the categorical pattern (ORZ hard zeros in backtracking/
uncertainty, init-vs-deduction split). profiles.json also has raw (unthresholded)
profiles that sum to 1.

EXPERIMENTS COMPLETE. Both figures' data in hand.

## Write-up (remaining)

Fig 1 (category profiles, ORZ vs OpenThinker), Fig 2 (steering curves: OT
within-model, OT->ORZ transfer, random control). State: fidelity rejected
(Exp 2), category divergence (Exp 1 + ORZ's zero backtracking/uncertainty),
confounds (base-vs-instruct, CoT format, norm scale), null-vs-signal framing.