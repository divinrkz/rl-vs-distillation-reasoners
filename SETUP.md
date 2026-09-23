# Running on Lightning AI

Target: one **L40S (48GB)** GPU. Develop with the GPU **off**; switch it on only
while a run executes — Lightning bills per GPU-hour and the project has a
20-hour budget.

## 1. Push this repo to GitHub (from your Mac)

```bash
cd /Users/divinirakiza/Workspaces/recherche/rl-distill
git init && git add . && git commit -m "Step 1: smoke test scaffold"
# create an empty repo named rl-distill on github.com, then:
git remote add origin https://github.com/divinrkz/rl-distill.git
git branch -M main && git push -u origin main
```

`externals/` and `experiments/` are gitignored on purpose (upstream forks are
cloned on the box; results never get committed).

## 2. Create the Studio

1. New Studio on lightning.ai. Start on the default **CPU** machine.
2. Open the terminal and clone:
   ```bash
   git clone https://github.com/divinrkz/rl-distill.git
   cd rl-distill
   ```

## 3. Bootstrap (CPU is fine for this step)

```bash
bash scripts/bootstrap.sh
```

Installs `requirements.txt` and clones the two upstream forks at their pinned
commits. Weights download to `$HF_HOME` (persistent Studio storage), so keep the
GPU off until this finishes.

If either model or the dataset is gated, set a token first:
```bash
export HF_TOKEN=hf_...
```

## 4. Switch to the L40S and run the smoke test

Change the Studio machine to **L40S**, then:

```bash
python scripts/01_smoke.py configs/smoke.yaml
```

Writes `experiments/smoke/{orz,openthinker}/results.json` plus a config copy.
The vLLM settings in `configs/smoke.yaml` (`gpu_memory_utilization: 0.90`,
`max_model_len: 20000`) are sized for 48GB with a 7B model and have headroom.

## 5. Turn the GPU back off

Switch the Studio back to CPU (or stop it) as soon as the run finishes. Outputs
persist on Studio storage regardless of machine type.

## Notes

- One model is loaded at a time and freed before the next, so a single 48GB card
  handles both 7B models sequentially.
- The Qwen2.5-32B labeler (Steps 3+) needs more than 48GB in bf16 — plan to run
  it AWQ/GPTQ-quantized on the L40S, or on an 80GB card. Deferred until then.
- To pull local edits onto the box later: `git pull` in the Studio.

