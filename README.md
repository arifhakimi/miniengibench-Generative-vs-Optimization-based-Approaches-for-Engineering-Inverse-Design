# Mini-EngiBench: A Lightweight Benchmark for ML in Engineering Design

I'm using this repo as a compact, reproducible playground for comparing a few tabular ML baselines against a tiny neural net on engineering-flavoured datasets. The goal is to show that I can set up clean pipelines, keep runs reproducible, and explain the trade-offs I'm seeing.

## Why I built it
- I wanted an end-to-end benchmark I can run quickly when I talk about engineering design ML.
- The IDEAL lab at ETH keeps asking for tidy infrastructure, so I've packaged datasets, models, metrics, and plots the way I like to work.
- Everything is small on purpose: it's easy to rerun, inspect, and extend without spinning up heavy tooling.

None of the datasets require deep physics knowledge, just solid ML hygiene.

## What's inside
- A reproducible pipeline around two regression tasks:
  - **Airfoil Self-Noise**: predict sound pressure level from airfoil and flow descriptors.
  - **Concrete Compressive Strength**: predict strength from mix proportions.
- Standardised train/val/test splits with RMSE, MAE, R^2, and timing metrics.
- JSON logs plus plotting helpers for quick comparisons.
- A conditional-VAE generative design benchmark (`benchmarks/run_generative_design.py`) that
  logs validity, diversity, and surrogate-comparison diagnostics.
- An inverse-design loop (`inverse_design.py`) that lets me test surrogate-driven optimisation under different data budgets.

## Environment setup (tested on Apple Silicon)
1. Install Miniforge: https://github.com/conda-forge/miniforge
2. Create and activate the environment:
```bash
conda create -n mini-engibench python=3.11 -y
conda activate mini-engibench
```
3. Pull in the scientific stack:
```bash
conda install -c conda-forge numpy pandas scipy scikit-learn matplotlib jupyterlab -y
conda install -c conda-forge xgboost openpyxl -y
```
4. Optional: install PyTorch (CPU wheel works fine; MPS is a bonus if available).
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

If I stick to pip, I run `pip install -r requirements.txt` inside the environment above.

## Project structure
```
mini_engibench/
  datasets/           # dataset loaders with a load_<task>() helper
  models/             # wrappers exposing fit/predict
  benchmarks/         # scripts orchestrating experiments and logging metrics
  results/            # JSON outputs and figures
  notebooks/          # scratchpads or mini walk-throughs
```

Keeping loaders, models, and experiment scripts in separate folders makes it painless to add new pieces without touching everything else.

## Datasets I'm using
- **Airfoil Self-Noise (UCI)**: small aerospace regression dataset that runs quickly.
- **Concrete Compressive Strength (UCI)**: civil engineering flavour with different target behaviour.

Both are public, tidy, and easy to standardise, so I can focus on the benchmarking side.
Each loader also accepts `return_metadata=True` to expose the scaler and feature bounds needed for generative validity checks.

## How I run things
1. Run the airfoil benchmark:
```bash
python -m benchmarks.run_airfoil
```
   This writes metrics and timings to `results/airfoil.json`.

2. Run the concrete benchmark:
```bash
python -m benchmarks.run_concrete
```
   Outputs land in `results/concrete.json`.

3. Regenerate the comparison plots:
```bash
python -m benchmarks.plot_results --task airfoil
python -m benchmarks.plot_results --task concrete
```
   Figures show up as `results/fig_airfoil.png` and `results/fig_concrete.png`.

4. Check how models behave with less data:
```bash
python -m benchmarks.run_data_efficiency --task airfoil
python -m benchmarks.run_data_efficiency --task concrete
```
   Each run logs `results/<task>_data_efficiency.json` for 10%, 20%, and 50% training splits.

   Plotting helper:
```bash
python -m benchmarks.plot_data_efficiency --task airfoil --split test
python -m benchmarks.plot_data_efficiency --task concrete --split test
```
   Swap `--split val` if I only care about validation scores. The figures land under `results/<task>_data_eff_*.png`.

5. Spin up the inverse-design loop:
```bash
python -m benchmarks.run_inverse_design --task airfoil --budget 25 --num-runs 10
python -m benchmarks.run_inverse_design --task concrete --budget 25 --num-runs 10
```
   This logs optimisation traces to `results/<task>_inverse_design.json` for different starting data budgets.

   Plot the optimisation traces:
```bash
python -m benchmarks.plot_inverse_design --task airfoil --quantity best
python -m benchmarks.plot_inverse_design --task concrete --quantity regret
```
   Saved figures follow `results/<task>_inverse_design_<quantity>.png`.

6. Sample designs with the conditional VAE benchmark:
```bash
python -m benchmarks.run_generative_design --task airfoil --num-samples 256 --eval-surrogate
python -m benchmarks.run_generative_design --task concrete --num-samples 256 --eval-surrogate
```
   The script saves `results/<task>_generative_cvae.json` with full training config, validity/diversity stats,
   surrogate score summaries, and (when `--eval-surrogate` is used) a histogram under
   `results/<task>_generative_performance_hist.png`.

## How I read the results
- MAE / RMSE: lower is better.
- R^2: closer to 1 means more variance explained.
- Train / inference timing: I keep an eye on these when thinking about optimisation loops.

## Generative design outputs
Each CVAE run writes `results/<task>_generative_cvae.json` with:
- `validity`: counts of valid/invalid samples plus per-feature constraint violations.
- `valid_design_indices`: zero-based indices for the samples that stayed within the scaled training envelope.
- `diversity`: average pairwise distance among valid unscaled designs when metadata is available.
- `performance`: surrogate statistics on the valid set and the histogram path (if `--eval-surrogate`).
- `paper1_comparison`: best inverse-design baseline recovered from `results/<task>_inverse_design.json` to compare against.

When the surrogate is evaluated, the script also saves `results/<task>_generative_performance_hist.png` to visualise the score distribution.

## Extending the benchmark
- New dataset: drop a loader in `datasets/` with a `load_<name>()` helper that returns train/val/test splits and, when `return_metadata=True`, the fitted scaler plus min/max ranges for generative validity checks.
- New model: add a wrapper in `models/` exposing `fit` and `predict`, just like the existing ones.
- New task: create `benchmarks/run_<name>.py` that wires loaders and models together and dumps JSON the same way.

The consistent interface keeps any additions from breaking the rest of the pipeline.

## Notes for a short report
When I write things up, I usually cover: brief motivation, dataset summaries, models and metrics, a couple of plots from `plot_results`, observations on trade-offs, and the exact commands plus environment details for reproducibility.

## Troubleshooting on Apple Silicon
- If `xgboost` fails to import, reinstall it from `conda-forge` inside the environment above.
- If PyTorch cannot see MPS, I stick to CPU; the workloads are tiny.
- If UCI downloads hiccup because of SSL, I download the CSV manually and point the loader to it.

## Ideas I still want to explore
- Add a third engineering dataset (maybe something CFD-related).
- Wire in lightweight hyperparameter sweeps with Optuna or similar.
- Ship a `Dockerfile` or `environment.yml` for one-command setup.
- Publish the repo publicly and archive a release on Zenodo for a DOI.

That's the whole setup.
# miniengibench-benhmark-of-surrogates-model-inverse-design
# miniengibench-Generative-vs-Optimization-based-Approaches-for-Engineering-Inverse-Design
