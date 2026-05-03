# Changelog

## 2026-05-02 — Project restructure, output routing, and RNG reproducibility fix

Three independent improvements applied in this session.

### Project folder reorganisation (all files)

The two legacy numbered folders (`01.Production code and results/`,
`02.Visualization code/`) have been replaced with a clean directory hierarchy:

```
src/training/      — training notebook, pipeline functions, ADASYN module
src/visualization/ — visualization notebook, plotting helpers
data/input/        — input dataset CSV
outputs/models/    — JLD2 results files
outputs/metrics/   — per-fold and inner-fold metric CSVs
outputs/predictions/ — subject-level prediction CSVs
outputs/shap/      — SHAP value CSVs
outputs/figures/   — PDF and PNG figures
outputs/tables/    — manuscript table CSVs
assets/            — static assets
```

All `include` paths, data load paths, JLD2 save/load paths, and `CairoMakie.save`
/ `savefig` paths in both notebooks have been updated to reflect the new layout.

### functions_opt_grid_v9.jl — `output_dir` keyword for export functions

Four export functions that previously wrote to `pwd()` unconditionally now accept
an `output_dir::String = "."` keyword argument. The training notebook passes the
appropriate `outputs/` subdirectory at each call site so all generated files land
in the correct folder automatically.

Functions updated:
- `export_all_results_to_csv`
- `export_baseline_sex_stratified`
- `export_baseline_sex_stratified_shap`
- `generate_supplemental_table`

### functions_opt_grid_v9.jl — RNG fix in `predict_with_tuned_neural`

**Problem**: The `training_losses` capture block in `predict_with_tuned_neural`
always took the else-branch (MLJTuning on the installed version does not expose
`model_report` in `best_report`). The else-branch called `MLJ.fit!` from an
unpredictable global RNG state, advancing it by an indeterminate amount and
breaking fixed-seed reproducibility for any subsequent global-RNG consumer.

**Fix**: `Random.seed!(rng)` is called immediately before the fallback refit,
where `rng` is the same `Int` seed already passed into `predict_with_tuned_neural`.
This makes the refit deterministic — the same seed always advances global RNG by
the same amount — without changing any classifier's own seeded behaviour.

Two diagnostic `println` statements added during debugging were also removed.

---

## 2026-05-02

Sex-stratified inner-fold metrics are now included in the cross-validation
transparency output, matching the granularity of the outer CV results.

### functions_opt_grid_v9.jl

- **`extract_inner_fold_metrics` — `sex_map` keyword argument**: New optional
  keyword `sex_map::Dict = Dict()` accepts a mapping of dataset name to sex
  vector (e.g. `Dict("baseline" => sex_vector_b)`). When provided for the
  baseline dataset, the function generates `baseline_male` and
  `baseline_female` rows for both inner-validation and outer-test splits,
  mirroring the sex-stratified breakdown produced by the main training loop.

- **New private helper `_push_sex_val_rows!`**: Splits inner-val predictions
  by sex (1.0 = male, 0.0 = female) and appends `baseline_male` /
  `baseline_female` rows to the results DataFrame. Skips any sex-group that
  has fewer than 2 samples or only one class label.

- **Baseline outer-test sex rows**: The outer-test block now reads saved
  `<clf>_sex` keys from `all_results` for the baseline dataset and pushes the
  corresponding `baseline_male` / `baseline_female` test-split rows, so the
  inner-fold DataFrame covers sex-stratified performance at every split level.

### PD_Training_2LyCV_SHAP_Grid_MultiSt_ClassImb_v7.jl

- **Skip checkbox default changed to off**: `CheckBox(default=true)` is now
  `CheckBox(default=false)`. The "Skip if already computed" toggle in the
  Inner Fold Metric Extraction section defaults to **unchecked**, so the
  second-pass extraction always re-runs unless the user explicitly enables the
  skip guard.

- **`sex_map` passed to `extract_inner_fold_metrics`**: The Apply-button cell
  now passes `sex_map = Dict("baseline" => sex_vector_b)` to ensure
  sex-stratified inner metrics are generated for the baseline dataset.

---

## 2026-04-30

All changes in this release address a peer reviewer request for cross-validation
transparency: inner-fold training and validation performance is now captured,
exported, and visualized alongside outer test results.

### functions_opt_grid_v9.jl

- **`evaluate_neural_network` — training loss return**: Function signature
  changed from returning a single `EvaluationResult` to a `(result, training_losses)`
  tuple. `training_losses` is the per-epoch loss vector retrieved from
  `report(mach).training_losses` after MLJFlux training.

- **`evaluate_neural_network_tuned` — training loss return**: Same change as
  above; `predict_with_tuned_neural` now surfaces the best-model epoch-loss
  vector, which is returned as the second element of the tuple.

- **New function `extract_inner_fold_metrics`**: Re-runs inner CV splits
  post-hoc to recover per-inner-fold train/val metrics (all classifiers, all
  N candidates) without repeating feature selection or hyperparameter search.
  Appends outer test metrics for direct comparison. Returns a flat DataFrame
  with columns Dataset, Method, N_Features, Classifier, Outer_Fold, Inner_Fold,
  Split, MCC, F1, Sensitivity, Specificity, BACC, Accuracy.

- **New function `generate_supplemental_table`**: Aggregates validation rows
  from `extract_inner_fold_metrics` into mean ± 95% CI (t-distribution, df = n−1)
  for every configuration, prints via PrettyTables, and saves a timestamped CSV.

### visual functions_v4.jl

- **New function `generate_table2`**: Formats inner-fold validation metrics
  (15 observations per best configuration) as mean (lower–upper) 95% CI using
  t(df=14) critical value; intended for manuscript Table 2.

- **New function `plot_inner_vs_outer_mcc`**: Produces Figure 6 — CairoMakie
  boxplots of inner train/val MCC overlaid with outer test mean ± 95% CI error
  bars, one panel per dataset × best configuration.

- **New function `plot_nn_loss_curves`**: Produces Figure 7 — 2×3 CairoMakie
  grid showing per-fold epoch-level training loss curves for NeuralNetwork and
  NeuralNetwork_TUNED, plus an aggregate panel.

### PD_Training_2LyCV_SHAP_Grid_MultiSt_ClassImb_v7.jl

- **NN training loss capture in main training loop**: At the end of the outer
  fold loop, `nn_training_losses` is assembled as a nested Dict
  `[dataset][fold][method][feat_key]` and saved to JLD2 alongside
  `all_results_adasyn`.

- **New "Inner Fold Metric Extraction (Second Pass)" section**: Post-hoc Pluto
  section with dataset multi-select, a Skip-if-already-computed checkbox
  (default on), and an Apply button. Calls `extract_inner_fold_metrics`, appends
  `inner_fold_metrics_adasyn` to the existing JLD2, and exports a timestamped
  CSV. The skip logic checks for the `inner_fold_metrics_adasyn` key in the JLD2
  before running so re-clicking is safe.

- **New JLD2 keys**: `nn_training_losses` (saved in main training cell) and
  `inner_fold_metrics_adasyn` (appended in second-pass cell) are now present
  in results files produced on or after 2026-04-30.

### PD_Visuals_ADSYN_models_v5.jl

- **New "Supplemental Outputs: Table 2, Figures 6–8" section**: Loads
  `inner_fold_metrics_adasyn` and `nn_training_losses` from the JLD2 (with
  graceful `@warn` skips if keys are absent) and generates Table 2, Figure 6,
  Figure 7, and Figure 8.

- **Table 2** (`Table2_inner_fold_summary.csv`): Inner CV validation metrics
  (mean, 95% CI) for the best model configuration per dataset, generated via
  `generate_table2`.

- **Figure 6** (`Figure6_InnerOuter_MCC_<dataset>.png/pdf`, 4 files): Inner
  train/val vs outer test MCC comparison per dataset, via
  `plot_inner_vs_outer_mcc`.

- **Figure 7** (`Figure7_NN_loss_<dataset>.png/pdf`, 4 files): Epoch-level NN
  training loss curves per dataset, via `plot_nn_loss_curves`.

- **Figure 8** (`Figure8_ValMCC_vs_N_<dataset>.png/pdf`, 4 files): Inner
  validation MCC vs number of selected features across all N candidates, one
  panel per dataset.
