# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Julia research codebase for a manuscript on **Parkinson's disease (PD) diagnosis via voice features**. It implements a nested cross-validation (nested CV) ML pipeline with sex-stratified analysis, ADASYN-based class imbalance handling, and SHAP explainability. The primary output is a set of CSV results and publication figures.

## Repository Structure

```
01.Production code and results/    # Core ML pipeline + output CSV/JLD2 files
    ADASYN_v1.jl                   # ADASYN module (standalone, importable)
    functions_opt_grid_v9.jl       # All ML pipeline utilities and functions
    Parkinsons_Speech-Features.csv # Input dataset (753 features + label)
    results_ADASYN_*.jld2          # Serialized nested CV results (JLD2 format)
    results_*.csv                  # Exported per-fold metric tables

02.Visualization code/             # Pluto.jl notebook + visual helper functions
    PD_Visuals_ADSYN_models_v5.jl  # Main Pluto notebook for all figures/tables
    visual functions_v4.jl         # Plotting helpers (sourced by the notebook)
    *.csv / *.pdf / *.png          # Pre-generated figures and data exports
```

## Running the Code

**Julia version and threads**: Launch Julia with multiple threads for ADASYN parallelization:
```bash
julia --threads auto
```

**Production pipeline** (`functions_opt_grid_v9.jl`): This file contains only function definitions — it must be `include`d from a driver script or REPL session. It expects `ADASYN_v1.jl` to be included first:
```julia
include("ADASYN_v1.jl")
using .ADASYN
include("functions_opt_grid_v9.jl")
```

**Visualization notebook** (Pluto.jl): Open `PD_Visuals_ADSYN_models_v5.jl` in Pluto. It loads the pre-computed `results_ADASYN_*.jld2` file and sources `visual functions_v4.jl`.
```julia
using Pluto; Pluto.run()
```

**Run ADASYN tests**:
```bash
julia --threads auto ADASYN_v1.jl
```

## Architecture

### Datasets

Four datasets are derived from `Parkinsons_Speech-Features.csv` and stored in the JLD2 results:
- `baseline` (`Xb`, `yb`): All 753 features, both sexes
- `sex aware` (`Xs`, `yb`): 753 features + binary sex column (754 total)
- `male` (`Xm`, `ym`): Male subjects only
- `female` (`Xf`, `yf`): Female subjects only

Feature index ranges in baseline: 1–21 Baseline acoustic, 22–24 Intensity, 25–28 Formant Frequency, 29–32 Bandwidth, 33–55 Vocal Fold, 56–138 MFCC, 139–320 Wavelet, 321–752 TQWT, 753 Sex.

### Pipeline (functions_opt_grid_v9.jl)

The nested CV orchestrator runs outer × inner fold loops. Key stages per fold:
1. **Stratified subject-level split** (`stratified_subject_level_cv`): Keeps all recordings from one subject in the same split to prevent data leakage.
2. **ADASYN oversampling** (`resample_training_data → apply_adasyn`): Applied only to training folds; β=1.0, K=5.
3. **Standardization** (`standardize_pair`): Fit on train, applied to test.
4. **Feature selection**: mRMR (`mrmr_pearson`), Boruta (`run_boruta`), Pearson correlation. Consensus across inner folds via `consensus_voting`.
5. **Classification**: RandomForest, NeuralNetwork (Flux/MLJFlux), NaiveBayes, LogisticRegression, AdaBoost, SVM-RBF — each with optional tuned variants.
6. **Sex-stratified evaluation**: `apply_sex_splitting!` post-processes baseline fold results to produce `baseline_male` / `baseline_female` metrics without retraining.
7. **SHAP**: `compute_fold_shap` uses `ShapML.jl` on a per-fold RF model.
8. **NN training loss capture**: `evaluate_neural_network` and `evaluate_neural_network_tuned` now return a `(result, training_losses)` tuple. `training_losses` is the per-epoch loss vector from `report(mach).training_losses`. The orchestrator stores these under `"NeuralNetwork_losses"` / `"NeuralNetwork_TUNED_losses"` keys in `fold_results`, then post-processes them into the `nn_training_losses` dict for JLD2 export.

#### Post-processing / export functions (new in v9)

- `extract_inner_fold_metrics(all_results, dataset_map, top_n_feats_candidates, n_outer_folds, n_inner_folds, random_seed; sex_map=Dict()) → DataFrame`  
  Re-runs the inner-fold splits (no new grid search or feature selection) to recover per-inner-fold train and val metrics for every outer fold × method × N × classifier combination. Also appends outer test metrics (Split = `"test"`) for direct comparison. When `sex_map` contains a `"baseline"` entry (sex vector), the function additionally generates `baseline_male` and `baseline_female` rows for both inner-val and outer-test splits. Returns a flat DataFrame with 13 columns: Dataset, Method, N_Features, Classifier, Outer_Fold, Inner_Fold, Split, MCC, F1, Sensitivity, Specificity, BACC, Accuracy.

- `generate_supplemental_table(inner_fold_df, strategy="ADASYN") → DataFrame`  
  Filters Split == `"val"` rows from `extract_inner_fold_metrics` output, groups by Dataset × Method × N_Features × Classifier, and formats mean ± 95% CI (t-distribution, df = n−1) for all six metrics. Prints via PrettyTables and saves a timestamped CSV (`supplemental_inner_val_metrics_<strategy>_<timestamp>.csv`).

### Results storage

Nested CV results are persisted as a JLD2 file with key `all_results_adasyn`. The dict is structured:
```
all_results[dataset][outer_fold_idx][method]["N_features"][classifier] → NamedTuple(mcc, f1, sen, spec, bacc, acc, y_true, y_pred, y_prob)
```
Tuned classifiers include a `best_model` field. Sex-split results are stored under `"<classifier>_sex"` keys.

Additional keys saved to the same JLD2:
- `nn_training_losses` — nested Dict `[dataset][outer_fold_idx][method][feat_key]` → Dict with `"NeuralNetwork_losses"` and/or `"NeuralNetwork_TUNED_losses"` vectors. Saved together with `all_results_adasyn` in the main training cell.
- `inner_fold_metrics_adasyn` — flat DataFrame (columns: Dataset, Method, N_Features, Classifier, Outer_Fold, Inner_Fold, Split, MCC, F1, Sensitivity, Specificity, BACC, Accuracy). Appended to the same JLD2 in a second pass. Split values: `"train"`, `"val"`, `"test"`.

### Visualization (PD_Visuals_ADSYN_models_v5.jl)

Pluto notebook that loads the JLD2, sources `visual functions_v4.jl`, and generates all manuscript figures (bar charts, Jaccard index plots, confusion matrices, SHAP heatmaps). Figures are saved as both PDF and PNG.

#### New functions in visual functions_v4.jl (v4)

- `generate_table2(inner_df, best_configs) → DataFrame`  
  Filters Split == `"val"`, aggregates 15 inner-fold observations per best configuration (one per dataset), and formats mean (lower–upper) 95% CI using t(df=14, α=0.025). Prints via PrettyTables and returns a DataFrame. Used for manuscript Table 2.

- `plot_inner_vs_outer_mcc(inner_df, dataset_name, method, n_features; classifiers, clf_colors) → Figure`  
  Figure 6. CairoMakie figure with boxplots of inner-train (gray) and inner-val (steelblue) MCC per classifier, overlaid with outer-test mean ± 95% CI error bars (black). One figure per dataset × best configuration.

- `plot_nn_loss_curves(nn_losses, dataset_name, method, feat_key) → Figure`  
  Figure 7. 2×3 CairoMakie panel grid: five per-fold epoch-loss curves (NeuralNetwork and NeuralNetwork_TUNED overlaid) plus one aggregate panel. Shows training convergence across outer folds.

#### New supplemental outputs in PD_Visuals_ADSYN_models_v5.jl (v5)

The "Supplemental Outputs: Table 2, Figures 6–8" section loads `inner_fold_metrics_adasyn` and `nn_training_losses` from JLD2, then generates:
- **Table 2** (`Table2_inner_fold_summary.csv`) via `generate_table2`
- **Figure 6** (`Figure6_InnerOuter_MCC_<dataset>.png/pdf` × 4 datasets) via `plot_inner_vs_outer_mcc`
- **Figure 7** (`Figure7_NN_loss_<dataset>.png/pdf` × 4 datasets) via `plot_nn_loss_curves`
- **Figure 8** (`Figure8_ValMCC_vs_N_<dataset>.png/pdf` × 4 datasets) — val MCC vs N features across all N candidates

Each output cell skips gracefully with `@warn` if the required data key was not found in the loaded JLD2.

### Training Notebook (PD_Training_2LyCV_SHAP_Grid_MultiSt_ClassImb_v7.jl)

Pluto notebook that drives the full nested CV pipeline: loads the dataset, configures hyperparameters via interactive controls, runs the outer/inner CV loop, exports CSV and JLD2 results, computes SHAP values, and generates the supplemental tables.

#### Inner Fold Metric Extraction (Second Pass)

A dedicated Pluto section ("Inner Fold Metric Extraction (Second Pass)") runs `extract_inner_fold_metrics` post-hoc, reusing saved feature selections and best hyperparameters without repeating feature selection or grid search. Controls:
- Dataset multi-select (default: all four)
- **Skip if already computed** checkbox (default: off) — checks for the `inner_fold_metrics_adasyn` key in the current JLD2 before running; set it on to avoid re-extraction when the key already exists
- Apply button (`run_inner_extract`) triggers extraction; re-clicking is safe due to skip logic

Output is appended to the existing JLD2 under `inner_fold_metrics_adasyn` and also exported as a timestamped CSV (`inner_fold_metrics_ADASYN_<timestamp>.csv`).

## Key Dependencies

Core packages (no Project.toml — packages must be installed manually):
- `NearestNeighbors`, `Random`, `Statistics` — used by ADASYN
- `MLJ`, `MLJLIBSVMInterface`, `MLJDecisionTreeInterface`, `MLJFlux`, `MLJNaiveBayesInterface` — ML framework
- `DecisionTree`, `NaiveBayes`, `GLM`, `Flux` — algorithm backends
- `ShapML` — SHAP explainability
- `JLD2` — results serialization
- `CSV`, `DataFrames`, `ROCAnalysis`, `Distributions` — data handling
- `Plots`, `CairoMakie`, `StatsPlots`, `ColorSchemes` — visualization
- `Pluto`, `PlutoUI` — notebook runtime
- `PrettyTables`, `Crayons` — terminal output formatting

## Conventions

- Labels: `0` = Healthy Control (HC), `1` = Parkinson's Disease (PD)
- Sex encoding: `1.0` = male, `0.0` = female
- Primary metric: **MCC** (Matthews Correlation Coefficient); secondary metrics: F1, sensitivity, specificity, balanced accuracy
- Statistical tests: Cochran's Q (3+ methods), McNemar pairwise (2 methods)
- All random operations take an explicit `rng_seed::Int` for reproducibility; never rely on global RNG state in pipeline code
