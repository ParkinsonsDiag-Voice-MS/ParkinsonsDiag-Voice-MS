# Interpretable Voice-Based Parkinson's Detection: Sex-Specific Acoustic Biomarkers with Subject-Level Nested Cross-Validation

This repository accompanies the following publication:

> Briscoe, D. K., Castedo, R. V., Clarke, P., Di Stella, D., & Deng, J. (2025).
> "Interpretable Voice-Based Parkinson's Detection: Sex-Specific Acoustic Biomarkers
> with Subject-Level Nested Cross-Validation."
> Available at SSRN: https://ssrn.com/abstract=6430163

[![DOI](https://img.shields.io/badge/DOI-10.2139%2Fssrn.6430163-blue)](http://dx.doi.org/10.2139/ssrn.6430163)

## Citation
```bibtex
@article{briscoe2025parkinson,
  author    = {Briscoe, Dana K. and Castedo, Rolando Velasco and Clarke, Paul
               and {Di Stella}, Domenic and Deng, Jeremiah},
  title     = {Interpretable Voice-Based {Parkinson's} Detection: Sex-Specific
               Acoustic Biomarkers with Subject-Level Nested Cross-Validation},
  journal   = {SSRN},
  year      = {2025},
  doi       = {10.2139/ssrn.6430163},
  url       = {https://ssrn.com/abstract=6430163}
}
```

The project is implemented in **Julia** and organised around **Pluto.jl notebooks**, enabling fully reproducible results.

---

## Project Objectives

The main objectives of this project are:

- To evaluate **voice-based machine learning models** for Parkinson’s disease diagnosis.
- To investigate whether **sex-specific and sex-aware models** improve diagnostic performance compared to pooled (sex-agnostic) models.
- To ensure **unbiased performance estimation** by:
  - Enforcing **subject-level data separation**, so that recordings from the same subject never appear in both training and test sets.
  - Using **two-layer (nested) cross-validation**, separating feature selection, hyperparameter tuning, and final evaluation.
- To analyse **feature relevance and stability** using multiple feature selection techniques and **SHAP value explanations**.

This work builds on and extends prior PD voice-analysis literature by explicitly addressing common methodological issues such as subject leakage and improper cross-validation.

---

## Key Methodological Features

### 1. Subject-Level Data Splitting
Each subject contributes multiple voice recordings.  
All cross-validation folds are generated **at the subject level**, preventing information leakage between training and testing sets.

### 2. Nested Cross-Validation (Two-Layer CV)
- **Outer loop**: Subject-level stratified cross-validation for unbiased model evaluation.
- **Inner loop**: Independent feature selection and hyperparameter tuning.
- Feature selection is performed **only on training data** within the inner loop.

### 3. Sex-Aware and Sex-Specific Modelling
The pipeline evaluates four dataset configurations:
- **Baseline**: All subjects pooled (no sex information).
- **Sex-aware**: Sex included explicitly as a feature.
- **Male-only**: Models trained and evaluated only on male subjects.
- **Female-only**: Models trained and evaluated only on female subjects.

Additionally, baseline models are **post-hoc stratified by sex** to compare evaluation-only sex effects without retraining.

### 4. Feature Extraction and Selection
Feature sets are derived from voice recordings and include:
- Baseline dysphonia features
- Intensity and formant features
- MFCCs
- Wavelet features
- Tunable Q-factor Wavelet Transform (TQWT)

Feature selection methods:
- **mRMR**
- **Boruta**
- **Consensus voting** across inner folds

### 5. Class Imbalance Handling
- **ADASYN (Adaptive Synthetic Sampling)**  implementation created for this project as no Julia package at the time had ADASYN.
- ADASYN is applied **only to training data** within each fold.
- No synthetic samples ever leak into validation or test sets.

### 6. Models Evaluated
The pipeline evaluates multiple classifiers, including:
- Random Forest (tuned and untuned)
- Support Vector Machines (RBF kernel)
- Neural Networks (Flux / MLJ)
- Logistic Regression
- Naive Bayes
- AdaBoost

Performance metrics include:
- MCC (primary metric)
- F1-score
- Sensitivity
- Specificity
- Balanced accuracy
- ROC-AUC
---

## Requirements

Julia ≥ 1.9

Pluto.jl

Key Julia packages include:
- MLJ, MLJFlux, LIBSVM, DecisionTree
- Flux, ShapML
- CSV, DataFrames, StatsBase
- ROCAnalysis, Imbalance
- Plots, StatsPlots

All package dependencies are loaded directly inside the Pluto notebooks.

## How to Run

1. Install Julia ,follow  instructions at https://julialang.org/downloads/

2. Run Julia and Install Pluto: 

	```julia
		using Pkg
		Pkg.add("Pluto")
	```

3. Launch Pluto
	
	```Julia
	using Pluto; Pluto.run()
	```

4. Download and copy all the files of the repository in the same folder structure to your Julia working directory

5. Run the Model training notebook:
   
	a. From Pluto open  the Model training notebook (primary):

	```julia
	..\01.Production code and results\PD_Training_2LyCV_SHAP_Grid_MultiSt_ClassImb_v7.jl
	```
  
	b. Run the notebook so all dependencies are loaded and installed, the notebook will start with 'quick start' parameters for a fast 
	   1st run
   
	c. Change the configuration paramters to the recommended settings to match the paper results:

		-  Tick checkbox "Generate files" to Export results to csv and jl2d files
   		-  Nr of Outer Folds: 5
		-  Nr of Inner Folds: 3
		-  Top Features Candidates: 100
		-  Grid Resolution: 7 (grid search 7x7)
		-  Nr Models for AdaBoost: 30
	
	This files will be generated and must be copied to the visualization folder:

   		-  results_ADASYN_yyyymmdd_HHMMSS.jld2
		-  subject_predictions_detailed.csv
		-  subject_level_data_yyyymmdd_HHMMSS
		-  results_baseline_(ADASYN)_yyyymmdd_HHMMSS.csv
		-  results_baseline_female_(ADASYN)_yyyymmdd_HHMMSS.csv
		-  results_baseline_male_(ADASYN)_yyyymmdd_HHMMSS.csv
		-  results_female_(ADASYN)_yyyymmdd_HHMMSS.csv
		-  results_male_(ADASYN)_yyyymmdd_HHMMSS.csv
		-  results_sex aware_(ADASYN)_yyyymmdd_HHMMSS.csv
		-  shap_baseline_sex_stratified_ADASYN_yyyymmdd_HHMMSS.csv

7. open the visualization notebook and update the names of the files to be loaded with ones genetared during your run:
	```julia
	..\02.Visualization code\PD_Visuals_ADSYN_models_v5.jl
	```
## Dataset

https://www.kaggle.com/datasets/porinitahoque/parkinsons-disease-pd-data-analysis/data

Sakar, C.O., Serbes, G., Gunduz, A., Tunc, H.C., Nizam, H., Sakar, B.E., Tutuncu, M., Aydin, T., Isenkul, M.E. and Apaydin, H., 2018. A comparative analysis of speech signal processing algorithms for Parkinson's disease classification and the use of the tunable Q-factor wavelet transform. Applied Soft Computing, DOI: [Web Link] https://doi.org/10.1016/j.asoc.2018.10.022
	
##  Reproducibility Notes

All random processes are seeded.

Subject-level fold assignments are deterministic.

Feature selection, resampling, and tuning are fully nested.

No test data are used during feature selection or oversampling.

##  Disclaimer

This code is intended for research purposes only.
It is not a clinical diagnostic tool and should not be used for medical decision-making.

##  Citation

If you use or adapt this code, please cite the associated publication: To be added upon approval


