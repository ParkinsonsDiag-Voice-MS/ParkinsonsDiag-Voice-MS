
# ============================================================================
# TYPE DEFINITIONS
# ============================================================================
#=
# Standard evaluation result type (no best_model)
const EvaluationResult = NamedTuple{
    (:mcc, :f1, :sen, :spec, :bacc, :acc, :y_true, :y_pred, :y_prob),
    Tuple{Float64, Float64, Float64, Float64, Float64, Float64, 
          Vector{Int}, Vector{Int}, Vector{Float64}}}

# Tuned model result type (includes best_model)
const TunedEvaluationResult = NamedTuple{
    (:mcc, :f1, :sen, :spec, :bacc, :acc, :y_true, :y_pred, :y_prob, :best_model),
    Tuple{Float64, Float64, Float64, Float64, Float64, Float64, 
          Vector{Int}, Vector{Int}, Vector{Float64}, Any}}
=#

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
# Function to categorize a feature by its index.
# Pass the feature index to the function and it will return the feature's category as a string
# ============================================================================
function categorize_feature_by_index(idx::Int)
	if 1 <= idx <= 21
        return "Baseline"
    elseif 22 <= idx <= 24
        return "Intensity"
    elseif 25 <= idx <= 28
        return "Formant Frequency"
    elseif 29 <= idx <= 32
        return "Bandwidth"
    elseif 33 <= idx <= 55
        return "Vocal Fold"
    elseif 56 <= idx <= 138
        return "MFCC"
    elseif 139 <= idx <= 320
        return "Wavelet"
    elseif 321 <= idx <= 752
        return "TQWT"
    elseif idx == 753
        return "Sex"
    else
        return "Unknown"
    end
end

# ============================================================================
# Standardization
# ============================================================================
function standardize_pair(Xtr::AbstractMatrix{<:Real}, Xte::AbstractMatrix{<:Real})::Tuple{Matrix{Float64}, Matrix{Float64}}
    
    μ = mean(Xtr, dims=1)
    σ = std(Xtr, dims=1)
    
    # Avoid division by zero
    @inbounds for i in eachindex(σ)
        if σ[i] == 0.0
            σ[i] = 1.0
        end
    end
    
    return ((Xtr .- μ) ./ σ, (Xte .- μ) ./ σ)
end
# ============================================================================
# CLASS IMBALANCE HANDLING
# ============================================================================


# ============================================================================
# Stratified k-fold CV
# ============================================================================
function stratified_subject_level_cv(subjects::AbstractVector,
									 labels::AbstractVector{<:Integer},
									 k_folds::Int; random_seed::Int=42)
	@assert length(subjects) == length(labels)
	Random.seed!(random_seed)
	
	uniq_subj = unique(subjects)
	subj_label = Dict{eltype(uniq_subj),Int}()
	for s in uniq_subj
		i = findfirst(==(s), subjects)
		subj_label[s] = labels[i]
	end
	
	hc_subj = [s for s in uniq_subj if subj_label[s] == 0]
	pd_subj = [s for s in uniq_subj if subj_label[s] == 1]
	shuffle!(hc_subj); shuffle!(pd_subj)
	
	folds = Vector{NamedTuple{(:train,:test),Tuple{Vector{Int},Vector{Int}}}}()
	for i in 1:k_folds
		test_subj = vcat(hc_subj[i:k_folds:end], pd_subj[i:k_folds:end])
		test_idx  = findall(x -> x in test_subj, subjects)
		train_idx = setdiff(1:length(subjects), test_idx)
		push!(folds, (train=train_idx, test=test_idx))
	end
	folds
end



# ==========================================================================
# ADASYN WRAPPER - PRIMARY IMBALANCE HANDLING METHOD
# ===========================================================================
# Wrapper for ADASYN (Adaptive Synthetic Sampling) from ADASYN_v1.jl module.
# This function provides a consistent interface for use in the nested CV pipeline.
#
# Arguments:
#   - X: Feature matrix (samples × features)
#   - y: Label vector (binary: 0/1)
#   - rng_seed: Random seed for reproducibility
#
# Returns:
#   - X_resampled: Resampled feature matrix
#   - y_resampled: Resampled labels
# ============================================================================
function apply_adasyn(
    X::AbstractMatrix{<:Real}, 
    y::Vector{Int}; 
    rng_seed::Int
)::Tuple{Matrix{Float64}, Vector{Int}}
    
	# Create MersenneTwister from seed    
    # Use ADASYN module's fit_resample function
    # β=1.0 creates fully balanced dataset
    # K=5 is the number of nearest neighbors (standard default)
	# ADASYN module will create its own RNG internally
    rng = MersenneTwister(rng_seed)
    X_res, y_res = ADASYN.adasyn_fit_resample(
        X, y; 
        β=1.0,      # Full balance
        K=5,        # Number of neighbors
        d_th=0.75,  # Apply ADASYN if imbalance ratio < 0.75
        rng=rng
    )
    
    return (Matrix{Float64}(X_res), Vector{Int}(y_res))
end


# ============================================================================
# CENTRAL RESAMPLING DISPATCHER
# ============================================================================
# Applies ADASYN (Adaptive Synthetic Sampling) for class imbalance handling.
# This is the main entry point for all class imbalance handling in the pipeline.
#
# Arguments:
#   - X_train: Training feature matrix
#   - y_train: Training labels
#   - rng_seed: Random seed for reproducibility
#
# Returns:
#   - X_resampled: Resampled training features
#   - y_resampled: Resampled training labels
#
# Notes:
#   - Applied ONLY to training data, never to validation or test sets
# ============================================================================
function resample_training_data(
    X_train::AbstractMatrix{<:Real},
    y_train::Vector{Int};
    rng_seed::Int
)::Tuple{Matrix{Float64}, Vector{Int}}
    
    # Use ADASYN (Adaptive Synthetic Sampling) 
    return apply_adasyn(X_train, y_train; rng_seed=rng_seed)
end

# ============================================================================
# CROSS-VALIDATION
# ============================================================================


# ============================================================================
# Stratified subject-level k-fold cross-validation.
#   Ensures subjects are kept together and classes are balanced across folds.
#       NOTE: Uses Random.seed! for backward compatibility but should transition
#            to passing RNG objects in future versions.
# ============================================================================
function stratified_subject_level_cv( subjects::AbstractVector, labels::AbstractVector{<:Integer}, k_folds::Int;
    random_seed::Int=42)
    
    @assert length(subjects) == length(labels)
    Random.seed!(random_seed)
    
    uniq_subj = unique(subjects)
    subj_label = Dict{eltype(uniq_subj),Int}()
    for s in uniq_subj
        i = findfirst(==(s), subjects)
        subj_label[s] = labels[i]
    end
    
    hc_subj = [s for s in uniq_subj if subj_label[s] == 0]
    pd_subj = [s for s in uniq_subj if subj_label[s] == 1]
    shuffle!(hc_subj); shuffle!(pd_subj)
    
    folds = Vector{NamedTuple{(:train,:test),Tuple{Vector{Int},Vector{Int}}}}()
    for i in 1:k_folds
        test_subj = vcat(hc_subj[i:k_folds:end], pd_subj[i:k_folds:end])
        test_idx  = findall(x -> x in test_subj, subjects)
        train_idx = setdiff(1:length(subjects), test_idx)
        push!(folds, (train=train_idx, test=test_idx))
    end
    folds
end


# ============================================================================
#  Calculate Binary performance evaluation metrics
#  Returns: (mcc, f1, sensitivity, specificity, balanced_accuracy, accuracy)
# ============================================================================
function calc_perf_eval_measures( y_true::Vector{Int}, y_pred::Vector{Int})::Tuple{Float64, Float64, Float64, Float64, Float64, Float64}
    
    # Confusion matrix
    tp = sum((y_true .== 1) .& (y_pred .== 1))
    tn = sum((y_true .== 0) .& (y_pred .== 0))
    fp = sum((y_true .== 0) .& (y_pred .== 1))
    fn = sum((y_true .== 1) .& (y_pred .== 0))
    
    # Metrics (with zero-division protection)
    sen = tp == 0 && fn == 0 ? 0.0 : tp / (tp + fn)
    spec = tn == 0 && fp == 0 ? 0.0 : tn / (tn + fp)
    prec = tp == 0 && fp == 0 ? 0.0 : tp / (tp + fp)
    
    f1 = (prec == 0.0 && sen == 0.0) ? 0.0 : 2 * prec * sen / (prec + sen)
    bacc = (sen + spec) / 2.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    # MCC with numerical stability
    numerator = (tp * tn) - (fp * fn)
    denominator_squared = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = denominator_squared == 0 ? 0.0 : numerator / sqrt(Float64(denominator_squared))
    
    return (mcc, f1, sen, spec, bacc, acc)
end

# ============================================================================
# Calculate ROC-AUC Score
# ============================================================================
function roc_auc( scores::AbstractVector{<:Real}, y_true::AbstractVector{<:Integer}; positive_label::Integer=1)::Float64
	
    @assert length(scores) == length(y_true)
	pos = Float64.(scores[y_true .== positive_label])
	neg = Float64.(scores[y_true .!= positive_label])
	if isempty(pos) || isempty(neg)
		return NaN
	end
	ROCAnalysis.auc(ROCAnalysis.roc(pos, neg))
end

# ============================================================================
# Function and definition used for Majority vote
# ============================================================================

#shortcut for Majority vote
majority_vote(v) = mode(v)

#function for Majority vote , Type-stable majority
function majority_vote(preds::Vector{Int})::Int
    
    count_0 = sum(preds .== 0)
    count_1 = sum(preds .== 1)
    return count_1 > count_0 ? 1 : 0
end

# ============================================================================
# SEX-STRATIFIED EVALUATION FOR BASELINE MODELS
# ============================================================================
# These functions enable post-hoc splitting of baseline model predictions
# by sex without retraining. Only applicable to baseline dataset.
# ============================================================================

# ============================================================================
# Split Classifier Results by Sex (Subject-Level)
# ============================================================================
# Takes a classifier's evaluation result and sex information, splits
# predictions into male and female subsets, and computes metrics for each.
#
# Args:
#   result: NamedTuple from classifier evaluation (mcc, f1, ..., y_true, y_pred, y_prob)
#   sex_test: Sex vector for test subjects (1=male, 0=female), aligned with y_true
#
# Returns:
#   NamedTuple with :male and :female, each containing full metric sets
# ============================================================================
function split_result_by_sex(result::NamedTuple, sex_test::Vector{Float64})::NamedTuple{(:male, :female), Tuple{NamedTuple, NamedTuple}}
    
    # Extract predictions and labels
    y_true = result.y_true
    y_pred = result.y_pred
    y_prob = result.y_prob
    
    # Validate alignment
    @assert length(y_true) == length(sex_test) "Sex vector must align with predictions"
    
    # Create sex masks
    male_mask = sex_test .== 1.0
    female_mask = sex_test .== 0.0
    
    # Split predictions by sex
    y_true_male = y_true[male_mask]
    y_pred_male = y_pred[male_mask]
    y_prob_male = y_prob[male_mask]
    
    y_true_female = y_true[female_mask]
    y_pred_female = y_pred[female_mask]
    y_prob_female = y_prob[female_mask]
    
    # Compute metrics for male subset
    if length(y_true_male) > 0
        mcc_m, f1_m, sen_m, spec_m, bacc_m, acc_m = calc_perf_eval_measures(
            y_true_male, y_pred_male
        )
        male_result = (
            mcc=mcc_m, f1=f1_m, sen=sen_m, spec=spec_m, bacc=bacc_m, acc=acc_m,
            y_true=y_true_male, y_pred=y_pred_male, y_prob=y_prob_male
        )
    else
        # Empty result if no male subjects in test set
        male_result = (
            mcc=NaN, f1=NaN, sen=NaN, spec=NaN, bacc=NaN, acc=NaN,
            y_true=Int[], y_pred=Int[], y_prob=Float64[]
        )
    end
    
    # Compute metrics for female subset
    if length(y_true_female) > 0
        mcc_f, f1_f, sen_f, spec_f, bacc_f, acc_f = calc_perf_eval_measures(
            y_true_female, y_pred_female
        )
        female_result = (
            mcc=mcc_f, f1=f1_f, sen=sen_f, spec=spec_f, bacc=bacc_f, acc=acc_f,
            y_true=y_true_female, y_pred=y_pred_female, y_prob=y_prob_female
        )
    else
        # Empty result if no female subjects in test set
        female_result = (
            mcc=NaN, f1=NaN, sen=NaN, spec=NaN, bacc=NaN, acc=NaN,
            y_true=Int[], y_pred=Int[], y_prob=Float64[]
        )
    end
    
    return (male=male_result, female=female_result)
end
# ============================================================================
# Apply Sex Splitting to All Classifiers in Fold Results
# ============================================================================
# Post-processes fold results to add sex-stratified metrics for all classifiers
#
# Args:
#   fold_results: Results dictionary from process_outer_fold
#   sex_test: Sex vector for test subjects
#
# Modifies:
#   fold_results dictionary by adding "<classifier>_sex" keys
# ============================================================================
function apply_sex_splitting!(fold_results::Dict, sex_test::Vector{Float64})::Nothing
    
    # List of classifiers to split
    classifiers = [
        "RandomForest", "RandomForest_TUNED",
        "NeuralNetwork", "NeuralNetwork_TUNED",
        "NaiveBayes",
        "LogisticRegression",
        "AdaBoost", "AdaBoost_TUNED",
        "SVM_RBF", "SVM_RBF_TUNED"
    ]
    
    # Iterate through methods
    for method_name in ["mRMR", "Boruta"]
        for (key, value) in fold_results[method_name]
            # Skip selected_features entries
            if startswith(string(key), "selected_features")
                continue
            end
            
            # Process each classifier
            for classifier in classifiers
                if haskey(value, classifier)
                    result = value[classifier]
                    
                    # Split by sex and store with "_sex" suffix
                    sex_split = split_result_by_sex(result, sex_test)
                    value["$(classifier)_sex"] = sex_split
                end
            end
        end
    end
    
    return nothing
end
# ============================================================================
# Export Sex-Stratified Results to CSV
# ============================================================================
# Exports baseline results split by sex to separate CSV files
#
# Args:
#   baseline_results: Nested CV results for baseline dataset
#   top_n_feats_candidates: Feature counts tested
#   strategy: Imbalance strategy name (e.g., "RLOSamp")
#   sex_vector: Full sex vector aligned with subjects
#   outer_folds: Fold structure to extract test indices
# ============================================================================
function export_baseline_sex_stratified(
    baseline_results::Dict,
    top_n_feats_candidates::Vector{Int},
    strategy::String,
    sex_vector::Vector{Float64},
    outer_folds::Vector;
    output_dir::String = "."
)
    
    println("\n" * "="^60)
    println("Exporting sex-stratified baseline results...")
    println("="^60 * "\n")
    
    classifiers = [
        "RandomForest", "RandomForest_TUNED",
        "NeuralNetwork", "NeuralNetwork_TUNED",
        "NaiveBayes",
        "LogisticRegression",
        "AdaBoost", "AdaBoost_TUNED",
        "SVM_RBF", "SVM_RBF_TUNED"
    ]
    
    methods = ["mRMR", "Boruta"]
    
    # Initialize DataFrames for male and female
    male_df = DataFrame(
        Dataset = String[],
        Method = String[],
        N_Features = Int[],
        Classifier = String[],
        Fold = Int[],
        MCC = Float64[],
        F1 = Float64[],
        Sensitivity = Float64[],
        Specificity = Float64[],
        Balanced_Accuracy = Float64[],
        Accuracy = Float64[]
    )
    
    female_df = deepcopy(male_df)
    
    n_outer_folds = length(baseline_results)
    
    # Process each fold
    for outer_fold_idx in 1:n_outer_folds
        for method in methods
            for n_features in top_n_feats_candidates
                feature_count = "$(n_features)_features"
                
                for classifier in classifiers
                    # Check if sex-split results exist
                    sex_key = "$(classifier)_sex"
                    
                    if haskey(baseline_results[outer_fold_idx][method], feature_count) &&
                       haskey(baseline_results[outer_fold_idx][method][feature_count], sex_key)
                        
                        sex_results = baseline_results[outer_fold_idx][method][feature_count][sex_key]
                        
                        # Extract male results
                        if !isnan(sex_results.male.mcc)
                            push!(male_df, (
                                Dataset = "baseline_male",
                                Method = method,
                                N_Features = n_features,
                                Classifier = classifier,
                                Fold = outer_fold_idx,
                                MCC = sex_results.male.mcc,
                                F1 = sex_results.male.f1,
                                Sensitivity = sex_results.male.sen,
                                Specificity = sex_results.male.spec,
                                Balanced_Accuracy = sex_results.male.bacc,
                                Accuracy = sex_results.male.acc
                            ))
                        end
                        
                        # Extract female results
                        if !isnan(sex_results.female.mcc)
                            push!(female_df, (
                                Dataset = "baseline_female",
                                Method = method,
                                N_Features = n_features,
                                Classifier = classifier,
                                Fold = outer_fold_idx,
                                MCC = sex_results.female.mcc,
                                F1 = sex_results.female.f1,
                                Sensitivity = sex_results.female.sen,
                                Specificity = sex_results.female.spec,
                                Balanced_Accuracy = sex_results.female.bacc,
                                Accuracy = sex_results.female.acc
                            ))
                        end
                    end
                end
            end
        end
    end
    
    # Save to CSV
    date = Dates.format(now(), "yyyymmdd_HHMMSS")
    
    male_filename = joinpath(output_dir, "results_baseline_male_($strategy)_$date.csv")
    female_filename = joinpath(output_dir, "results_baseline_female_($strategy)_$date.csv")

    CSV.write(male_filename, male_df)
    CSV.write(female_filename, female_df)

    println("  ✓ Saved: $male_filename ($(nrow(male_df)) rows)")
    println("  ✓ Saved: $female_filename ($(nrow(female_df)) rows)")
    println("\n" * "="^60)
end


# ============================================================================
# FEATURE SELECTION
# ============================================================================

# ============================================================================
# Consensus Voting
#   Aggregate feature selections from multiple inner folds via majority voting
#   Returns consensus feature indices, ordered by frequency (most frequent first)
# ============================================================================
function consensus_voting(feature_lists::Vector, threshold::Int)::Vector{Int}
    # Count frequency of each feature and track first appearance order
    feature_frequency = Dict{Int, Int}()
    feature_first_rank = Dict{Int, Int}()
    
    for (fold_idx, feature_list) in enumerate(feature_lists)
        for (rank, feature) in enumerate(feature_list)
            # Update frequency
            feature_frequency[feature] = get(feature_frequency, feature, 0) + 1
            
            # Track earliest rank across folds (for ordering)
            if !haskey(feature_first_rank, feature)
                feature_first_rank[feature] = rank
            else
                # Use minimum rank seen across folds
                feature_first_rank[feature] = min(feature_first_rank[feature], rank)
            end
        end
    end
    
    # Keep features appearing >= threshold times
    consensus_features = [feat for (feat, freq) in feature_frequency 
                         if freq >= threshold]
    
    # Sort by frequency (descending), then by selection order (ascending rank)
    consensus_features = sort(consensus_features, 
                             by = feat -> (-feature_frequency[feat], 
                                          feature_first_rank[feat]))
    
    return consensus_features
end

# ============================================================================
# Function to display a single confusion matrix
# ============================================================================
function confusion_matrix(y_true::Vector, y_pred::Vector, label::String)
    
    # Calculate confusion matrix
    tn = sum((y_true .== 0) .& (y_pred .== 0))
    fp = sum((y_true .== 0) .& (y_pred .== 1))
    fn = sum((y_true .== 1) .& (y_pred .== 0))
    tp = sum((y_true .== 1) .& (y_pred .== 1))
    
    println("\n$label")
    
    # Display as table
    cm_df = DataFrame(
        Actual = ["HC", "PD"],
        Predicted_HC = [tn, fn],
        Predicted_PD = [fp, tp]
    )
    
    pretty_table(cm_df,
                 header = ["Actual", "Predicted HC", "Predicted PD"],
                 alignment = [:l, :r, :r],
                 crop = :none)
    
    total = tn + fp + fn + tp
    println("Total: $total")
end
# ============================================================================
# STATISTICAL TESTING FUNCTIONS
# ============================================================================

# ============================================================================
# Cochran's Q for k>2 related classifiers (binary correct/incorrect per subject)
# ============================================================================
function cochrans_q(y_true::AbstractVector{<:Integer}, Ŷ::AbstractMatrix{<:Integer})
	n, k = size(Ŷ)
    M = Array{Int}(undef, n, k)             # correctness matrix (1=correct, 0=wrong)
    @inbounds for i in 1:n, j in 1:k
        M[i,j] = (Ŷ[i,j] == y_true[i]) ? 1 : 0
    end
    Cj = vec(sum(M, dims=1))                # correct counts per method
    Ri = vec(sum(M, dims=2))                # per-subject totals
    T  = sum(Cj)
    num = (k - 1) * (k * sum(Cj .^ 2) - T^2)
    den = k * sum(Ri) - sum(Ri .^ 2)
    Q   = den == 0 ? 0.0 : num / den        # guard against degenerate case
    p   = 1 - cdf(Chisq(k - 1), Q)
    return (Q=Q, df=k-1, p_value=p, significant=(p < 0.05))
end

# ============================================================================
# Perform Cochran's Q test comparing feature selection methods across all classifiers using nested CV results.
# Args:
#        nested_cv_results: Dictionary from nested CV orchestrator
#        top_n_feats_candidates: Vector of feature counts (e.g., [25, 50])
#		 dataset_name: char string with the name of the dataset being used.
#        n_features: (Optional) Specific feature count to test. If nothing, uses maximum.
# ============================================================================
function nested_cv_cochrans_q_test(nested_cv_results, top_n_feats_candidates; 
                                   dataset_name::String, 
                                   n_features::Union{Int,Nothing} = nothing)
    
    classifiers = [
        "RandomForest", "RandomForest_TUNED",
        "NeuralNetwork", "NeuralNetwork_TUNED",
        "NaiveBayes",
        "LogisticRegression",
        "AdaBoost", "AdaBoost_TUNED",
        "SVM_RBF", "SVM_RBF_TUNED"]
    classifier_display_names = [
        "RF", "RF (Tuned)",
        "NeuralNet", "NeuralNet (Tuned)",
        "NaiveBayes",
        "LogReg",
        "AdaBoost", "AdaBoost (Tuned)",
        "SVM-RBF", "SVM-RBF (Tuned)"]
    
    methods = ["mRMR", "Boruta"]
    
    # Use provided n_features or default to maximum
    n_features_selected = isnothing(n_features) ? maximum(top_n_feats_candidates) : n_features
    feature_count = "$(n_features_selected)_features"
    
    # Initialize results storage
    Qvals = Float64[]
    pvals = Float64[]
    sigs = Bool[]
    
    # Perform Cochran's Q test for each classifier
    for (clf_name, clf_display) in zip(classifiers, classifier_display_names)
        
        # Aggregate predictions across all outer folds and methods
		y_true_all = Int[]
		predictions_matrix = Vector{Int}[]
        
        for method_name in methods
            method_preds = []
            
            for outer_fold_idx in 1:length(nested_cv_results)
                metrics = nested_cv_results[outer_fold_idx][method_name][feature_count][clf_name]
                
                # Store true labels from first method to avoid duplication
                if method_name == "mRMR"
                    append!(y_true_all, metrics.y_true)
                end
                
                append!(method_preds, metrics.y_pred)
            end
            
            push!(predictions_matrix, method_preds)
        end
        
        # Convert to matrix format (subjects × methods)
        Ŷ = hcat(predictions_matrix...)
        
        # Perform Cochran's Q test
        result = cochrans_q(y_true_all, Ŷ)
        
        push!(Qvals, result.Q)
        push!(pvals, result.p_value)
        push!(sigs, result.significant)
    end
    
    # Create summary DataFrame
    cochran_summary = DataFrame(
        Classifier = classifier_display_names,
        Q = round.(Qvals, digits=4),
        df = fill(2, length(classifiers)),
        p_value = round.(pvals, digits=4),
        significant = sigs
    )
    
    # Row highlighter: highlight if significant or p_value ≤ 0.05
    hl_row = Highlighter(
        (data, i, j) -> (data[i, 5] === true) || (data[i, 4] isa Real && data[i, 4] <= 0.05),
        Crayon(background=:light_yellow, foreground=:black, bold=false)
    )
    
    # Display title
    println("="^75)
    println("Cochran's Q Test $dataset_name: Feature Selection Methods Across Classifiers")
    println("(Using $n_features_selected Features)")
    println("="^75)
    println()
    
    # Display table
    pretty_table(cochran_summary,
                 header = ["Classifier", "Q", "df", "p_value", "significant"],
                 formatters = ft_printf("%.3f", [2, 4]),
                 alignment = [:l, :r, :r, :r, :r],
                 columns_width = [18, 8, 8, 10, 12],
                 autowrap = true,
                 crop = :none,
                 highlighters = (hl_row,),
                 show_subheader = false)
    
    println()
end

# ============================================================================
# Function to perform McNemar Test
# ============================================================================
function mcnemar_test(y_true::Vector{Int}, y_pred1::Vector{Int}, y_pred2::Vector{Int})
    # Contingency table: e01 = method1 wrong & method2 correct, e10 = opposite
    e01 = sum((y_pred1 .!= y_true) .& (y_pred2 .== y_true))
    e10 = sum((y_pred1 .== y_true) .& (y_pred2 .!= y_true))
    
    # Chi-square with continuity correction
    χ² = (abs(e01 - e10) - 1)^2 / (e01 + e10)
    p_value = 1 - cdf(Chisq(1), χ²)
    
    return (p_value=p_value, significant=(p_value < 0.05))
end

# ============================================================================
# Perform pairwise McNemar tests comparing feature selection methods across classifiers.
#    Args:
#          nested_cv_results: Dictionary from nested CV orchestrator
#          top_n_feats_candidates: Vector of feature counts (e.g., [25, 50])
#		   dataset_name: char string with the name of the dataset being used.
#          n_features: (Optional) Specific feature count to test. If nothing, uses maximum.
# ============================================================================
function nested_cv_mcnemar_pairwise(nested_cv_results, top_n_feats_candidates; 
                                   dataset_name::String, 
                                   n_features::Union{Int,Nothing} = nothing)
    
    classifiers = ["RandomForest","NeuralNetwork", "NaiveBayes", 
                   "LogisticRegression", "AdaBoost", "SVM_RBF"]
    classifier_display_names = ["Random Forests", "Neural Network", "Naive Bayes",
                                "Logistic Regression", "AdaBoost", "SVM-RBF"]
    clf_abbrev = ["RF", "NN", "NB", "LR", "Ada", "SVM"]
    
    methods = ["mRMR", "Boruta"]
   # Use provided n_features or default to maximum
    n_features_selected = isnothing(n_features) ? maximum(top_n_feats_candidates) : n_features
    feature_count = "$(n_features_selected)_features"

    
    # Generate all pairwise combinations of methods
    comps = [(a, b, "$a vs $b") for (a, b) in combinations(methods, 2)]
    
    # Initialize DataFrame with comparison names as first column
    summary_tbl = DataFrame(Comparison = [name for (_, _, name) in comps])
    
    # Loop through each classifier and compute McNemar p-values
    for (clf_name, clf_display, clf_short) in zip(classifiers, classifier_display_names, clf_abbrev)
        pcol = Float64[]  # Vector to store p-values for this classifier
        
        # Aggregate predictions across all outer folds for each method
        method_predictions = Dict{String, Vector{Int}}()
        y_true_all = Int[]
        
        for method_name in methods
            method_preds = Int[]
            
            for outer_fold_idx in 1:length(nested_cv_results)
                metrics = nested_cv_results[outer_fold_idx][method_name][feature_count][clf_name]
                
                # Store true labels from first method to avoid duplication
                if method_name == "mRMR"
                    append!(y_true_all, metrics.y_true)
                end
                
                append!(method_preds, metrics.y_pred)
            end
            
            method_predictions[method_name] = method_preds
        end
        
        # For each pairwise comparison, compute McNemar's test
        for (method_a, method_b, _) in comps
            y_pred_a = method_predictions[method_a]
            y_pred_b = method_predictions[method_b]
            
            p_value = mcnemar_test(y_true_all, y_pred_a, y_pred_b).p_value
            push!(pcol, round(p_value, digits = 4))
        end
        
        # Add this classifier's p-values as a new column
        summary_tbl[!, clf_short] = pcol
    end
    
    # Highlighter: highlight p-values ≤ 0.05
    hl = Highlighter(
        (data, i, j) -> j in 2:7 && data[i, j] isa Real && !ismissing(data[i, j]) && 
                        data[i, j] <= 0.05,
        Crayon(background = :light_yellow, foreground = :black, bold = false)
    )
    
    # Display title
    println("="^75)
    println("McNemar p-values $dataset_name: Pairwise Feature Selection Method Comparisons")
    println("(Top $n_features_selected Features)")
    println("="^75)
    println()
    
    # Display table
    pretty_table(summary_tbl,
                 header        = ["Comparison", clf_abbrev...],
                 formatters    = ft_printf("%.3f", 2:7),
                 alignment     = [:l, :r, :r, :r, :r, :r, :r],
                 columns_width = [20, 5, 5, 5, 5, 5, 5],
                 autowrap      = true,
                 crop          = :none,
                 highlighters  = (hl,))
    
    println("\np-values ≤ 0.05 are highlighted")
    println()
end
# ============================================================================
# Perform pairwise McNemar tests for TUNED classifiers only.
# Same structure as nested_cv_mcnemar_pairwise, but using tuned models in a separate table.
#		   dataset_name: char string with the name of the dataset being used.
#          n_features: (Optional) Specific feature count to test. If nothing, uses maximum.
# ============================================================================

function nested_cv_mcnemar_pairwise_tuned(nested_cv_results, top_n_feats_candidates; dataset_name::String, 
                                   n_features::Union{Int,Nothing} = nothing)
    
    tuned_classifiers = ["RandomForest_TUNED",
                         "NeuralNetwork_TUNED",
                         "AdaBoost_TUNED",
                         "SVM_RBF_TUNED"]
    
    tuned_display_names = ["Random Forests (Tuned)",
                           "Neural Network (Tuned)",
                           "AdaBoost (Tuned)",
                           "SVM-RBF (Tuned)"]
    
    # Short headers for the columns
    tuned_abbrev = ["RF_T", "NN_T", "Ada_T", "SVM_T"]
    
    methods        = ["mRMR", "Boruta"]
    # Use provided n_features or default to maximum
    n_features_selected = isnothing(n_features) ? maximum(top_n_feats_candidates) : n_features
    feature_count = "$(n_features_selected)_features"

    
    # All pairwise combinations of feature selection methods
    comps = [(a, b, "$a vs $b") for (a, b) in combinations(methods, 2)]
    
    # First column is the comparison name
    summary_tbl = DataFrame(Comparison = [name for (_, _, name) in comps])
    
    # Loop through tuned classifiers
    for (clf_name, clf_display, clf_short) in zip(tuned_classifiers,
                                                  tuned_display_names,
                                                  tuned_abbrev)
        pcol = Float64[]
        method_predictions = Dict{String, Vector{Int}}()
        y_true_all = Int[]
        
        # Aggregate predictions across outer folds for each FS method
        for method_name in methods
            method_preds = Int[]
            
            for outer_fold_idx in 1:length(nested_cv_results)
                metrics = nested_cv_results[outer_fold_idx][method_name][feature_count][clf_name]
                
                if method_name == "mRMR"
                    append!(y_true_all, metrics.y_true)
                end
                
                append!(method_preds, metrics.y_pred)
            end
            
            method_predictions[method_name] = method_preds
        end
        
        # Pairwise McNemar tests between FS methods for this tuned classifier
        for (method_a, method_b, _) in comps
            y_pred_a = method_predictions[method_a]
            y_pred_b = method_predictions[method_b]
            
            p_value = mcnemar_test(y_true_all, y_pred_a, y_pred_b).p_value
            push!(pcol, round(p_value, digits = 4))
        end
        
        summary_tbl[!, clf_short] = pcol
    end
    
    # Now we have 1 + 4 columns = 5 total
    hl = Highlighter(
        (data, i, j) -> j in 2:5 && data[i, j] isa Real && !ismissing(data[i, j]) &&
                        data[i, j] <= 0.05,
        Crayon(background = :light_yellow, foreground = :black, bold = false)
    )
    
    println("="^75)
    println("McNemar p-values $dataset_name: Pairwise Feat Selection Method Comparisons ")
    println("TUNED models (Top $n_features_selected)")
    println("="^75)
    println()
    
    pretty_table(summary_tbl,
                 header        = ["Comparison", tuned_abbrev...],
                 formatters    = ft_printf("%.3f", 2:5),
                 alignment     = [:l, :r, :r, :r, :r],
                 columns_width = [20, 7, 7, 7, 7],
                 autowrap      = true,
                 crop          = :none,
                 highlighters  = (hl,))
    
    println("\np-values ≤ 0.05 are highlighted")
    println()
end

function export_baseline_sex_stratified_shap(
    all_datasets_shap::Dict,
    top_n_feats_candidates::Vector{Int},
    strategy::String;
    output_dir::String = "."
)

    println("\n" * "="^60)
    println("Exporting baseline sex-stratified SHAP values...")
    println("="^60 * "\n")

    rows = DataFrame(
        Dataset = String[],
        Method = String[],
        N_Features = Int[],
        Feature = String[],
        MeanAbsSHAP = Float64[],
        StdAbsSHAP = Float64[]
    )

    baseline_shap = all_datasets_shap["baseline"]

    for (method, method_dict) in baseline_shap
        for n_features in top_n_feats_candidates
            for shap_label in ["baseline", "baseline_male", "baseline_female"]

                key = (n_features, shap_label)
                haskey(method_dict, key) || continue

                shap_df = method_dict[key]

                for row in eachrow(shap_df)
                    push!(rows, (
                        Dataset = shap_label,
                        Method = method,
                        N_Features = n_features,
                        Feature = row.feature_name,
                        MeanAbsSHAP = row.mean_abs_shap,
                        StdAbsSHAP = row.std_abs_shap
                    ))
                end
            end
        end
    end

    date = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename = joinpath(output_dir, "shap_baseline_sex_stratified_$(strategy)_$date.csv")
    CSV.write(filename, rows)

    println("✓ Saved: $filename ($(nrow(rows)) rows)")
    println("\n" * "="^60)
end

# ===================================================================================
# Compute SHAP values for a single outer fold.
#
# Supports:
#   - Standard SHAP (all datasets)
#   - Baseline sex-stratified SHAP (baseline, baseline_male, baseline_female)
#
# Returns:
#   - Dict{String, DataFrame} mapping dataset_label => SHAP DataFrame
# ===================================================================================

function compute_fold_shap(
    fold_idx::Int,
    X::Matrix{Float64},
    y::Vector{Int},
    selected_features::Vector{Int},
    feature_names::Vector{String},
    outer_folds,
    sample_size::Int,
    random_seed::Int;
    sex_vector::Union{Vector{Float64},Nothing} = nothing
)::Dict{String,DataFrame}

    # -----------------------------
    # Fold indices
    # -----------------------------
    outer_fold = outer_folds[fold_idx]
    train_idx = outer_fold.train
    test_idx  = outer_fold.test

    # -----------------------------
    # Train / test split
    # -----------------------------
    X_train = X[train_idx, selected_features]
    y_train = y[train_idx]
    X_test  = X[test_idx, selected_features]

    # Standardisation (train → test)
    X_train_sc, X_test_sc = standardize_pair(X_train, X_test)

    # -----------------------------
    # Train RF model
    # -----------------------------
    fold_rng = StableRNG(random_seed + fold_idx)
    rf_model = DecisionTree.RandomForestClassifier(
        n_trees = 100,
        rng = fold_rng
    )
    DecisionTree.fit!(rf_model, X_train_sc, y_train)

    selected_feature_names = feature_names[selected_features]

    # -----------------------------
    # Determine explanation groups
    # -----------------------------
    explain_groups = Dict{String, Vector{Bool}}()

    if isnothing(sex_vector)
        # Normal SHAP (no sex split)
        explain_groups["default"] = trues(size(X_test_sc, 1))
    else
        sex_test = sex_vector[test_idx]

        explain_groups["baseline"]        = trues(length(sex_test))
        explain_groups["baseline_male"]   = sex_test .== 1
        explain_groups["baseline_female"] = sex_test .== 0
    end

    # -----------------------------
    # Compute SHAP per group
    # -----------------------------
    shap_results = Dict{String,DataFrame}()

    for (label, mask) in explain_groups
        sum(mask) == 0 && continue  # safety

        explain_df = DataFrame(
            X_test_sc[mask, :],
            selected_feature_names
        )

        shap_df = ShapML.shap(
            explain = explain_df,
            reference = explain_df,
            model = rf_model,
            predict_function = predict_function,
            sample_size = sample_size,
            seed = random_seed + fold_idx
        )

        shap_df.fold = fill(fold_idx, nrow(shap_df))
        shap_df.dataset = fill(label, nrow(shap_df))

        shap_results[label] = shap_df
    end

    return shap_results
end

# ============================================================================
# Calculate Jaccard indices between feature selection methods
# ============================================================================
function jaccard_index(set1::Vector{Int}, set2::Vector{Int})::Float64
    intersection = length(intersect(set1, set2))
    union_size = length(union(set1, set2))
    return union_size == 0 ? 0.0 : intersection / union_size
end

# Helper for SHAP
function predict_function(model, data)
    data_matrix = Matrix(data)
    probs = DecisionTree.predict_proba(model, data_matrix)
    return DataFrame(y_pred = probs[:, 2])
end


# ==================================================
# Fast mRMR approximation using Pearson correlation, Maximizes relevance to target while 
# minimizing redundancy between features.
#   Optimized with:
#    - Type-stable return
#    - Pre-allocated arrays
#    - Vectorized operations where possible
#    - Using views to avoid copies
# ==================================================

function mrmr_feature_selection(X::Matrix{<:Real}, y::Vector{Int}, n_features::Int)::Vector{Int}

    n_total = size(X, 2)
    selected_feat_indices = Vector{Int}(undef, 0)
    sizehint!(selected_feat_indices, n_features)
    remaining = collect(1:n_total)
    
    # Pre-compute ALL correlations with target (cached)
    target_cors = Vector{Float64}(undef, n_total)
    @inbounds for i in 1:n_total
        target_cors[i] = abs(cor(view(X, :, i), y))
    end
    
    # First feature: maximum relevance
    best_idx = argmax(target_cors[remaining])
    push!(selected_feat_indices, remaining[best_idx])
    deleteat!(remaining, best_idx)
    
    # Pre-allocate scores array (reused each iteration)
    max_remaining = length(remaining)
    scores = Vector{Float64}(undef, max_remaining)
    
    # Iterative selection: maximize (relevance - redundancy)
    for iter in 2:n_features
        isempty(remaining) && break
        
        n_rem = length(remaining)
        
        # Compute mRMR scores for all remaining features
        @inbounds for idx in 1:n_rem
            i = remaining[idx]
            relevance = target_cors[i]
            
            # Compute mean redundancy with selected features
            redundancy_sum = 0.0
            n_selected = length(selected_feat_indices)
            for j in selected_feat_indices
                redundancy_sum += abs(cor(view(X, :, i), view(X, :, j)))
            end
            redundancy = redundancy_sum / n_selected
            
            scores[idx] = relevance - redundancy
        end
        
        # Select feature with highest mRMR score
        best_idx = argmax(view(scores, 1:n_rem))
        push!(selected_feat_indices, remaining[best_idx])
        deleteat!(remaining, best_idx)
    end
    
    return selected_feat_indices
end

# =====================================================================
# Boruta feature selection algorithm
# Wrapper method using Random Forests to identify all relevant features
# by comparing them against randomized "shadow" features
# =====================================================================

	# Recursively count feature usage in decision tree
	# Traverses tree and increments importance counter for each feature used in splits
function count_features_b!(imp::Vector{Float64}, node)
    node isa DecisionTree.Leaf && return
    imp[node.featid] += 1.0
    isdefined(node, :left) && !isnothing(node.left) && count_features_b!(imp, node.left)
    isdefined(node, :right) && !isnothing(node.right) && count_features_b!(imp, node.right)
end
# ============================================================================
# Train Random Forests and compute feature importances
# ============================================================================
function get_importances(X::Matrix{<:Real}, y::Vector{Int}, n_trees::Int, max_depth::Int, rng)::Vector{Float64}
    n_feat = size(X, 2)
    # Build Random Forest with sqrt(n_features) features per split
    model_b = build_forest(y, X, max(1, Int(floor(sqrt(n_feat)))), n_trees, 0.7, max_depth, 1, 2, 0.0; rng=rng)
    # Count how often each feature is used across all trees
    imp = zeros(Float64, n_feat)
    for tree in model_b.trees
        count_features_b!(imp, tree)  
    end
    # Normalize to sum to 1
    return imp ./ sum(imp)
end
# ============================================================================
# Main Boruta algorithm
# Compares real features against "shadow" features (randomized copies)
# Confirms features that consistently outperform their shadows
# ============================================================================
function boruta_feature_selection(X::Matrix{<:Real}, y::Vector{Int}, n_features::Int; n_trees::Int=100, max_depth=5, max_iter=150,
    alpha::Real=0.05, perc=90, two_step=false, seed=42, verbose=false)::Vector{Int}
    
    # Initialize variables
    rng = MersenneTwister(seed)
    n_samp, n_feat = size(X)
    confirmed, rejected, hits = falses(n_feat), falses(n_feat), zeros(Int, n_feat)
    sum_imp = zeros(Float64, n_feat)
    
    # Iterative testing
    for iter in 1:max_iter
        # Check if all features have been decided
        undecided = .!(confirmed .| rejected)
        sum(undecided) == 0 && (verbose && println("✓ Done at iteration $iter"); break)
        
        # Create shadows and combine
        X_shadow = mapslices(col -> shuffle(rng, col), X, dims=1)
        X_both = hcat(X, X_shadow)
        
        # Get importances
        imp = get_importances(X_both, y, n_trees, max_depth, rng)
        orig_imp, shadow_imp = imp[1:n_feat], imp[n_feat+1:end]
        
        # Define threshold from shadow feature importances
        shadow_threshold = percentile(shadow_imp, perc)
        
        # Test each undecided feature
        for i in 1:n_feat
            undecided[i] || continue
            # Count as "hit" if feature beats shadow threshold
            orig_imp[i] > shadow_threshold && (hits[i] += 1)
            
            # Confirm features significantly better than random
            p_better = 1.0 - cdf(Binomial(iter, 0.5), hits[i] - 1)
            p_better < alpha && (confirmed[i] = true)
            
            # Reject features significantly worse than random
            if !two_step
                p_worse = cdf(Binomial(iter, 0.5), hits[i])
                p_worse < alpha && (rejected[i] = true)
            end
        end
        # Accumulate importances for final ranking
        sum_imp .+= orig_imp
    end
    
    # Reject remaining undecided features at the end if two_step
    if two_step
        undecided = .!(confirmed .| rejected)
        rejected[undecided] .= true
    end
    
    # Compute final ranking based on average importance 
    tentative = .!(confirmed .| rejected)
    ranking = invperm(sortperm(sum_imp ./ max_iter, rev=true))

    # Extract confirmed features and sort by importance rank
    conf_idx = findall(confirmed)
    sorted_idx = conf_idx[sortperm(ranking[conf_idx])]

    # Select top features (or all if fewer than n_features confirmed)
    selected_feat_indices = sorted_idx[1:min(n_features, length(sorted_idx))]

    return selected_feat_indices
end

# ============================================================================
# MAIN NESTED CV FUNCTION
# ============================================================================
#  Process Outer Fold
#   Process a single outer CV fold. This function is called in parallel by the main 
#       nested CV loop. Each fold gets its own deterministic RNG seed.
# Threading Safety
#   -No shared mutable state between threads
#   -Returns a complete Dict for this fold
#   -RNG seeded deterministically (base_seed + fold_idx)
# ============================================================================
function process_outer_fold( 
    outer_fold_idx::Int, 
    outer_fold::NamedTuple, 
    X::Matrix{Float64}, 
    y::Vector{Int},
    subjects::Vector{Int}, 
    feature_names::Vector{String}, 
    n_inner_folds::Int, 
    n_features_max::Int, 
    top_n_feats_candidates::Vector{Int},
    consensus_threshold::Int, 
    base_seed::Int, 
    resol::Int,
    resol1::Int,
    verbose::Bool;
    sex_vector::Union{Vector{Float64}, Nothing} = nothing  # Optional sex vector for baseline sex-stratification
    )::Dict
    
    # Create fold-specific RNG seed for reproducibility
    fold_seed = base_seed + outer_fold_idx
    
    # Initialize fold results
    fold_results = Dict()
    
    # Extract outer training and test indices
    outer_train_idx = outer_fold.train
    outer_test_idx = outer_fold.test
    
    # Extract outer training set
    X_outer_train = X[outer_train_idx, :]
    y_outer_train = y[outer_train_idx]
    subjects_outer_train = subjects[outer_train_idx]

    # Extract outer test set
    X_outer_test = X[outer_test_idx, :]
    y_outer_test = y[outer_test_idx]
    subjects_outer_test = subjects[outer_test_idx]
    
    # INNER LOOP: Feature Selection
    
    mrmr_inner_selected_all = []
    boruta_inner_selected_all = []
    
    # Get inner folds
    inner_folds = stratified_subject_level_cv(
        subjects_outer_train, 
        y_outer_train, 
        n_inner_folds; 
        random_seed=fold_seed
    )
    
    # Process inner folds
    for inner_fold_idx in 1:n_inner_folds
        inner_fold = inner_folds[inner_fold_idx]
        inner_train_idx = inner_fold.train
        
        X_inner_train = X_outer_train[inner_train_idx, :]
        y_inner_train = y_outer_train[inner_train_idx]
        
        # Feature selection with fold-specific seed
        inner_seed = fold_seed + inner_fold_idx
        
        mrmr_selected = mrmr_feature_selection(
            X_inner_train, y_inner_train, n_features_max)
        boruta_selected = boruta_feature_selection(
            X_inner_train, y_inner_train, n_features_max; seed=inner_seed)
        
        push!(mrmr_inner_selected_all, mrmr_selected)
        push!(boruta_inner_selected_all, boruta_selected)
    end
  
    # Aggregate via consensus voting
    mrmr_final = consensus_voting(mrmr_inner_selected_all, consensus_threshold)
    boruta_final = consensus_voting(boruta_inner_selected_all, consensus_threshold)
    
    # Create dictionaries for different feature counts
    mrmr_dict = Dict(n => mrmr_final[1:min(n, length(mrmr_final))] 
                    for n in top_n_feats_candidates)
    boruta_dict = Dict(n => boruta_final[1:min(n, length(boruta_final))] 
                      for n in top_n_feats_candidates)
    
    # EVALUATE CLASSIFIERS FOR EACH METHOD AND FEATURE COUNT
        
    # Initialize storage
    fold_results["mRMR"] = Dict()
    fold_results["Boruta"] = Dict()
    fold_results["jaccard_indices"] = Dict()
    
    # Store selected features
    for n in top_n_feats_candidates
        fold_results["mRMR"]["selected_features_$(n)"] = mrmr_dict[n]
        fold_results["Boruta"]["selected_features_$(n)"] = boruta_dict[n]
        
        # Calculate Jaccard indices
        fold_results["jaccard_indices"][n] = Dict(
            "mRMR_Boruta" => jaccard_index(mrmr_dict[n], boruta_dict[n])
        )
    end
    
    # Evaluate classifiers for each method and feature count
    for (method_name, features_dict) in [
        ("mRMR", mrmr_dict),
        ("Boruta", boruta_dict)
    ]
        
        for n in top_n_feats_candidates
            selected_features = features_dict[n]
            feature_count = "$(n)_features"
            
            # Subset data to selected features
            X_outer_train_sel = X_outer_train[:, selected_features]
            X_outer_test_sel = X_outer_test[:, selected_features]
            
            # Initialize storage for this method/feature combo
            fold_results[method_name][feature_count] = Dict()
            
            # Evaluate all classifiers
            fold_results[method_name][feature_count]["RandomForest"] = 
                evaluate_random_forests(
                    X_outer_train_sel, y_outer_train,
                    X_outer_test_sel, y_outer_test, subjects_outer_test;
                    train_subjects=subjects_outer_train,
                    rng_seed=fold_seed,
                )
            
            nn_result, nn_losses = evaluate_neural_network(
                    X_outer_train_sel, y_outer_train,
                    X_outer_test_sel, y_outer_test, subjects_outer_test;
                    train_subjects=subjects_outer_train,
                    rng_seed=fold_seed,
                )
            fold_results[method_name][feature_count]["NeuralNetwork"] = nn_result
            fold_results[method_name][feature_count]["NeuralNetwork_losses"] = nn_losses
            
            fold_results[method_name][feature_count]["NaiveBayes"] = 
                evaluate_naive_bayes(
                    X_outer_train_sel, y_outer_train,
                    X_outer_test_sel, y_outer_test, subjects_outer_test;
                    train_subjects=subjects_outer_train,
                    rng_seed=fold_seed,
                )
            
            fold_results[method_name][feature_count]["LogisticRegression"] = 
                evaluate_logistic_regression(
                    X_outer_train_sel, y_outer_train,
                    X_outer_test_sel, y_outer_test, subjects_outer_test;
                    train_subjects=subjects_outer_train,
                    rng_seed=fold_seed,
                )
            
            fold_results[method_name][feature_count]["AdaBoost"] = 
                evaluate_adaboost(
                    X_outer_train_sel, y_outer_train,
                    X_outer_test_sel, y_outer_test, subjects_outer_test;
                    train_subjects=subjects_outer_train,
                    rng_seed=fold_seed,
                )
            
            fold_results[method_name][feature_count]["SVM_RBF"] = 
                evaluate_svm_rbf(
                    X_outer_train_sel, y_outer_train,
                    X_outer_test_sel, y_outer_test, subjects_outer_test;
                    train_subjects=subjects_outer_train,
                    rng_seed=fold_seed,
                )
            
            # Tuned models
            fold_results[method_name][feature_count]["SVM_RBF_TUNED"] = 
                evaluate_svm_rbf_tuned(
                    X_outer_train_sel, y_outer_train,
                    X_outer_test_sel, y_outer_test, subjects_outer_test;
                    train_subjects=subjects_outer_train,
                    rng_seed=fold_seed, reso = resol
                )
            
            fold_results[method_name][feature_count]["RandomForest_TUNED"] = 
                evaluate_random_forests_tuned(
                    X_outer_train_sel, y_outer_train,
                    X_outer_test_sel, y_outer_test, subjects_outer_test;
                    train_subjects=subjects_outer_train,
                    rng_seed=fold_seed, reso = resol
                )
            
            fold_results[method_name][feature_count]["AdaBoost_TUNED"] = 
                evaluate_adaboost_tuned(
                    X_outer_train_sel, y_outer_train,
                    X_outer_test_sel, y_outer_test, subjects_outer_test;
                    train_subjects=subjects_outer_train,
                    rng_seed=fold_seed, reso = resol1
                )
            
            nn_tuned_result, nn_tuned_losses = evaluate_neural_network_tuned(
                    X_outer_train_sel, y_outer_train,
                    X_outer_test_sel, y_outer_test, subjects_outer_test;
                    train_subjects=subjects_outer_train,
                    rng_seed=fold_seed, reso = resol
                )
            fold_results[method_name][feature_count]["NeuralNetwork_TUNED"] = nn_tuned_result
            fold_results[method_name][feature_count]["NeuralNetwork_TUNED_losses"] = nn_tuned_losses
            
            if verbose
                println("Completed: $method_name | Outer Fold $outer_fold_idx | $feature_count")
                flush(stdout)
            end
        end
    end

    # Apply sex-stratified evaluation for baseline dataset (if sex_vector provided)
    if sex_vector !== nothing
        # Extract test indices for this fold
        outer_test_idx = outer_fold.test
        
        # Get sex information for test subjects (SUBJECT-LEVEL, not row-level)
        # subjects_outer_test has 3 rows per subject, but predictions are subject-level
        # So we need to get unique subjects and extract one sex value per subject
        
        unique_test_subjects = unique(subjects_outer_test)
        
        # Create subject-to-sex mapping
        subject_to_sex = Dict{Int, Float64}()
        for (idx, subj_id) in enumerate(subjects)
            if !haskey(subject_to_sex, subj_id)
                subject_to_sex[subj_id] = sex_vector[idx]
            end
        end
        
        # Extract sex for unique test subjects (in order they appear)
        sex_test = [subject_to_sex[subj_id] for subj_id in unique_test_subjects]
        
        if verbose
            println("    Fold $outer_fold_idx: $(length(unique_test_subjects)) test subjects ($(sum(sex_test .== 1.0)) male, $(sum(sex_test .== 0.0)) female)")
        end
        
        # Apply sex splitting to all classifier results
        apply_sex_splitting!(fold_results, sex_test)
        
        if verbose
            println("  ✓ Sex-stratified metrics computed for fold $outer_fold_idx")
            flush(stdout)
        end
    end

    return fold_results
end

# ============================================================================
# CLASSIFIER EVALUATION FUNCTIONS
# All evaluation functions support:
# - Subject-level oversampling (via oversample_subjects parameter)
# - Deterministic RNG seeding  
# - Class weights (where applicable)
# - Type-stable returns
# ============================================================================

# ============================================================================
# Define evaluate_single_classifier
#   Unified interface for evaluating any classifier
#    Useful for parallel evaluation of multiple classifiers
# ============================================================================
function evaluate_single_classifier( classifier_name::String, 
	X_train::AbstractMatrix{<:Real},
	y_train::Vector{Int},
    X_test::AbstractMatrix{<:Real},
	y_test::Vector{Int},
	test_subjects::Vector{Int},
	train_subjects::Vector{Int},
    rng_seed::Int=42,
)::EvaluationResult
    
    if classifier_name == "RandomForest"
        return evaluate_random_forests(X_train, y_train, X_test, y_test, test_subjects, rng_seed=rng_seed)
    elseif classifier_name == "NeuralNetwork"
        return evaluate_neural_network(X_train, y_train, X_test, y_test, test_subjects, rng_seed=rng_seed)
    elseif classifier_name == "NaiveBayes"
        return evaluate_naive_bayes(X_train, y_train, X_test, y_test, test_subjects, rng_seed=rng_seed)
    elseif classifier_name == "LogisticRegression"
        return evaluate_logistic_regression(X_train, y_train, X_test, y_test, test_subjects, rng_seed=rng_seed)
    elseif classifier_name == "AdaBoost"
        return evaluate_adaboost(X_train, y_train, X_test, y_test, test_subjects, rng_seed=rng_seed)
    elseif classifier_name == "SVM_RBF"
        return evaluate_svm_rbf(X_train, y_train, X_test, y_test, test_subjects, rng_seed=rng_seed)
    elseif classifier_name == "SVM_RBF_TUNED"
        return evaluate_svm_rbf_tuned(X_train, y_train, X_test, y_test, test_subjects; train_subjects=train_subjects, rng_seed=rng_seed)
    elseif classifier_name == "RandomForest_TUNED"
        return evaluate_random_forests_tuned(X_train, y_train, X_test, y_test, test_subjects; train_subjects=train_subjects, rng_seed=rng_seed)
    elseif classifier_name == "AdaBoost_TUNED"
        return evaluate_adaboost_tuned(X_train, y_train, X_test, y_test, test_subjects; train_subjects=train_subjects, rng_seed=rng_seed)
    elseif classifier_name == "NeuralNetwork_TUNED"
        return evaluate_neural_network_tuned(X_train, y_train, X_test, y_test, test_subjects; train_subjects=train_subjects, rng_seed=rng_seed)
    else
        error("Unknown classifier: $classifier_name")
    end
end

# ================================================================================
# Random Forests
#Optimized with:
#   - Type-stable return
#   - Optimized subject aggregation
#   - Pre-allocated containers
# ================================================================================
function evaluate_random_forests(X_train::AbstractMatrix{<:Real}, y_train::Vector{Int},
    X_test::AbstractMatrix{<:Real}, y_test::Vector{Int},test_subjects::Vector{Int};
    train_subjects::Vector{Int}=collect(1:length(y_train)), rng_seed::Int=42, n_trees::Int=100,
)::EvaluationResult
    
    # Create RNG for this evaluation
    rng = StableRNG(rng_seed)
    
    # Pre-allocate subject aggregation containers
    subj_preds = Dict{Int, Vector{Int}}()
    subj_probs = Dict{Int, Vector{Float64}}()
    subj_true = Dict{Int, Int}()
    
    # Standardize
    X_train_S, X_test_S = standardize_pair(X_train, X_test)
    
    # Balance training data, Apply ADASYN for class imbalance handling
    X_train_S_bal, y_train_bal = resample_training_data(
        X_train_S, y_train;
        
        rng_seed=rng_seed
    )
    
    # Train Random Forest
    rf = MLJDecisionTreeInterface.RandomForestClassifier(n_trees=n_trees, rng=rng_seed)
    mach = machine(rf, MLJ.table(X_train_S_bal), categorical(y_train_bal))
    MLJ.fit!(mach, verbosity=0)
    
    # Predict
    yprob = MLJ.predict(mach, MLJ.table(X_test_S))
    p_pos = [Float64(pdf(s, 1)) for s in yprob]
    yhat = [p >= 0.5 ? 1 : 0 for p in p_pos]
    
    # Subject aggregation
    @inbounds for (i, subj_id) in enumerate(test_subjects)
        push!(get!(()->Int[], subj_preds, subj_id), yhat[i])
        push!(get!(()->Float64[], subj_probs, subj_id), p_pos[i])
        subj_true[subj_id] = y_test[i]
    end
    
    # Subject-level majority vote
    unique_subjects = sort(collect(keys(subj_true)))
    n_subjects = length(unique_subjects)
    
    subject_level_true = Vector{Int}(undef, n_subjects)
    subject_level_pred = Vector{Int}(undef, n_subjects)
    subject_level_prob = Vector{Float64}(undef, n_subjects)
    
    @inbounds for (idx, subj_id) in enumerate(unique_subjects)
        subject_level_true[idx] = subj_true[subj_id]
        subject_level_pred[idx] = majority_vote(subj_preds[subj_id])
        subject_level_prob[idx] = mean(subj_probs[subj_id])
    end
    
    # Calculate metrics
    mcc, f1, sen, spec, bacc, acc = calc_perf_eval_measures(
        subject_level_true, subject_level_pred)
    
    return (mcc=mcc, f1=f1, sen=sen, spec=spec, bacc=bacc, acc=acc,
            y_true=subject_level_true, y_pred=subject_level_pred,
            y_prob=subject_level_prob)
end

# ============================================================================
# EVALUATE_NEURAL_NETWORK (Optimized)
#  Neural Network evaluation with optional subject-level oversampling.
#  Optimized  
#   - Type-stable return
#   - get!() pattern for subject aggregation
#   - Pre-allocated result vectors
# ============================================================================
function evaluate_neural_network( X_train::AbstractMatrix{<:Real}, y_train::Vector{Int},
    X_test::AbstractMatrix{<:Real}, y_test::Vector{Int},
    test_subjects::Vector{Int}; train_subjects::Vector{Int}=collect(1:length(y_train)),rng_seed::Int=42,
)
    
    # Create RNG
    rng = StableRNG(rng_seed)
    
    # Pre-allocate subject aggregation containers
    subj_preds = Dict{Int, Vector{Int}}()
    subj_probs = Dict{Int, Vector{Float64}}()
    subj_true = Dict{Int, Int}()
    
    # Standardize
    X_train_S, X_test_S = standardize_pair(X_train, X_test)
    
    # Balance training data, strategy-based resampling
    X_train_S_bal, y_train_bal = resample_training_data(
        X_train_S, y_train;
        
        rng_seed=rng_seed
    )
    
    # Build and train neural network
    n_features = size(X_train_S_bal, 2)
    model = nn_builder(n_features)
    mach = machine(model, MLJ.table(X_train_S_bal), categorical(y_train_bal))
    MLJ.fit!(mach, verbosity=0)
    training_losses = report(mach).training_losses

    # Predict
    yprob = MLJ.predict(mach, MLJ.table(X_test_S))
    p_pos = [Float64(pdf(s, 1)) for s in yprob]
    yhat = [p >= 0.5 ? 1 : 0 for p in p_pos]
    
    # Subject aggregation
    @inbounds for (i, subj_id) in enumerate(test_subjects)
        push!(get!(()->Int[], subj_preds, subj_id), yhat[i])
        push!(get!(()->Float64[], subj_probs, subj_id), p_pos[i])
        subj_true[subj_id] = y_test[i]
    end
    
    # Subject-level majority vote
    unique_subjects = sort(collect(keys(subj_true)))
    n_subjects = length(unique_subjects)
    
    subject_level_true = Vector{Int}(undef, n_subjects)
    subject_level_pred = Vector{Int}(undef, n_subjects)
    subject_level_prob = Vector{Float64}(undef, n_subjects)
    
    @inbounds for (idx, subj_id) in enumerate(unique_subjects)
        subject_level_true[idx] = subj_true[subj_id]
        subject_level_pred[idx] = majority_vote(subj_preds[subj_id])
        subject_level_prob[idx] = mean(subj_probs[subj_id])
    end
    
    # Calculate metrics
    mcc, f1, sen, spec, bacc, acc = calc_perf_eval_measures(
        subject_level_true, subject_level_pred)

    result = (mcc=mcc, f1=f1, sen=sen, spec=spec, bacc=bacc, acc=acc,
              y_true=subject_level_true, y_pred=subject_level_pred,
              y_prob=subject_level_prob)
    return result, training_losses
end

# ================================================================================
# Naive Bayes
#   Evaluate Naive Bayes classifier with optional subject-level oversampling.
# ================================================================================
function evaluate_naive_bayes( X_train::AbstractMatrix{<:Real}, y_train::Vector{Int}, X_test::AbstractMatrix{<:Real},
    y_test::Vector{Int}, test_subjects::Vector{Int}; train_subjects::Vector{Int}=collect(1:length(y_train)),
    rng_seed::Int=42)::EvaluationResult
    
    # Create RNG
    rng = StableRNG(rng_seed)
    
    # Pre-allocate subject aggregation containers
    subj_preds = Dict{Int, Vector{Int}}()
    subj_probs = Dict{Int, Vector{Float64}}()
    subj_true = Dict{Int, Int}()
    
    # Standardize
    X_train_S, X_test_S = standardize_pair(X_train, X_test)
    
    # Balance training data, strategy-based resampling
    X_train_S_bal, y_train_bal = resample_training_data(
        X_train_S, y_train;
        
        rng_seed=rng_seed
    )
    
    # Get unique classes and compute priors
    classes = unique(y_train_bal)
    n_features = size(X_train_S_bal, 2)
    
    class_priors = Dict{Int, Float64}()
    for c in classes
        class_priors[c] = sum(y_train_bal .== c) / length(y_train_bal)
    end
    
    # Compute means and stds for each class
    means = Dict{Int, Vector{Float64}}()
    stds = Dict{Int, Vector{Float64}}()
    for c in classes
        class_mask = y_train_bal .== c
        class_data = X_train_S_bal[class_mask, :]
        means[c] = vec(mean(class_data, dims=1))
        stds[c] = vec(std(class_data, dims=1))
        stds[c] .= max.(stds[c], 1e-10)  # Avoid division by zero
    end
    
    # Compute log probabilities for test data
    n_test = size(X_test_S, 1)
    probs_matrix = zeros(Float64, length(classes), n_test)
    
    @inbounds for i in 1:n_test
        x = X_test_S[i, :]
        for (j, c) in enumerate(classes)
            log_prob = log(class_priors[c])
            for f in 1:n_features
                μ = means[c][f]
                σ = stds[c][f]
                log_prob += -0.5 * log(2π * σ^2) - ((x[f] - μ)^2) / (2 * σ^2)
            end
            probs_matrix[j, i] = log_prob
        end
    end
    
    # Convert log probabilities to probabilities (softmax)
    @inbounds for i in 1:n_test
        max_log_prob = maximum(probs_matrix[:, i])
        probs_matrix[:, i] = exp.(probs_matrix[:, i] .- max_log_prob)
        probs_matrix[:, i] ./= sum(probs_matrix[:, i])
    end
    
    # Extract predictions
    pos_class_idx = findfirst(==(1), classes)
    p_pos = vec(probs_matrix[pos_class_idx, :])
    yhat = [p >= 0.5 ? 1 : 0 for p in p_pos]
    
    # Subject aggregation
    @inbounds for (i, subj_id) in enumerate(test_subjects)
        push!(get!(()->Int[], subj_preds, subj_id), yhat[i])
        push!(get!(()->Float64[], subj_probs, subj_id), p_pos[i])
        subj_true[subj_id] = y_test[i]
    end
    
    # Subject-level majority vote
    unique_subjects = sort(collect(keys(subj_true)))
    n_subjects = length(unique_subjects)
    
    subject_level_true = Vector{Int}(undef, n_subjects)
    subject_level_pred = Vector{Int}(undef, n_subjects)
    subject_level_prob = Vector{Float64}(undef, n_subjects)
    
    @inbounds for (idx, subj_id) in enumerate(unique_subjects)
        subject_level_true[idx] = subj_true[subj_id]
        subject_level_pred[idx] = majority_vote(subj_preds[subj_id])
        subject_level_prob[idx] = mean(subj_probs[subj_id])
    end
    
    # Calculate metrics
    mcc, f1, sen, spec, bacc, acc = calc_perf_eval_measures(
        subject_level_true, subject_level_pred)
    
    return (mcc=mcc, f1=f1, sen=sen, spec=spec, bacc=bacc, acc=acc,
            y_true=subject_level_true, y_pred=subject_level_pred,
            y_prob=subject_level_prob)
end

# ================================================================================
# Logistic Regression
#   Evaluate Logistic Regression classifier with optional subject-level oversampling.
# ================================================================================

function evaluate_logistic_regression( X_train::AbstractMatrix{<:Real}, y_train::Vector{Int},
    X_test::AbstractMatrix{<:Real}, y_test::Vector{Int}, test_subjects::Vector{Int};
    train_subjects::Vector{Int}=collect(1:length(y_train)), rng_seed::Int=42, 
)::EvaluationResult
    
    # Create RNG
    rng = StableRNG(rng_seed)
    
    # Pre-allocate subject aggregation containers
    subj_preds = Dict{Int, Vector{Int}}()
    subj_probs = Dict{Int, Vector{Float64}}()
    subj_true = Dict{Int, Int}()
    
    # Standardize
    X_train_S, X_test_S = standardize_pair(X_train, X_test)
    
    # Balance training data,strategy-based resampling
    X_train_S_bal, y_train_bal = resample_training_data(
        X_train_S, y_train;
        
        rng_seed=rng_seed
    )
    
    # Prepare DataFrames for GLM
    train_df = DataFrame(X_train_S_bal, :auto)
    train_df.y = y_train_bal
    formula = Term(:y) ~ sum(Term.(Symbol.(names(train_df, Not(:y)))))
    test_df = DataFrame(X_test_S, :auto)
    
    # Declare variables for predictions
    local p_pos::Vector{Float64}, yhat::Vector{Int}
    
    # Try to fit with increased iterations, fallback if convergence fails
    try
        model = glm(formula, train_df, Binomial(), LogitLink(); 
                   maxiter=100, atol=1e-6, rtol=1e-6)
        
        p_pos_raw = GLM.predict(model, test_df)
        p_pos = Float64.(coalesce.(p_pos_raw, 0.5))
        yhat = [p >= 0.5 ? 1 : 0 for p in p_pos]
        
    catch e
        # Fallback: predict based on training class distribution
        @warn "Logistic Regression convergence failed, using class prior fallback"
        class_prior = mean(y_train_bal)
        p_pos = fill(class_prior, size(X_test_S, 1))
        yhat = [p >= 0.5 ? 1 : 0 for p in p_pos]
    end
    
    # Subject aggregation
    @inbounds for (i, subj_id) in enumerate(test_subjects)
        push!(get!(()->Int[], subj_preds, subj_id), yhat[i])
        push!(get!(()->Float64[], subj_probs, subj_id), p_pos[i])
        subj_true[subj_id] = y_test[i]
    end
    
    # Subject-level majority vote
    unique_subjects = sort(collect(keys(subj_true)))
    n_subjects = length(unique_subjects)
    
    subject_level_true = Vector{Int}(undef, n_subjects)
    subject_level_pred = Vector{Int}(undef, n_subjects)
    subject_level_prob = Vector{Float64}(undef, n_subjects)
    
    @inbounds for (idx, subj_id) in enumerate(unique_subjects)
        subject_level_true[idx] = subj_true[subj_id]
        subject_level_pred[idx] = majority_vote(subj_preds[subj_id])
        subject_level_prob[idx] = mean(subj_probs[subj_id])
    end
    
    # Calculate metrics
    mcc, f1, sen, spec, bacc, acc = calc_perf_eval_measures(
        subject_level_true, subject_level_pred)
    
    return (mcc=mcc, f1=f1, sen=sen, spec=spec, bacc=bacc, acc=acc,
            y_true=subject_level_true, y_pred=subject_level_pred,
            y_prob=subject_level_prob)
end

# ================================================================================
# Evaluate AdaBoost
#  Evaluate AdaBoost classifier with optional subject-level oversampling.
#  Optimization:
#    - Type-stable return
#    - get!() pattern for subject aggregation
#    - Pre-allocated result vectors
# ================================================================================
function evaluate_adaboost( X_train::AbstractMatrix{<:Real}, y_train::Vector{Int},
    X_test::AbstractMatrix{<:Real}, y_test::Vector{Int}, test_subjects::Vector{Int};
    train_subjects::Vector{Int}=collect(1:length(y_train)), rng_seed::Int=42,
)::EvaluationResult
    
    # Create RNG
    rng = StableRNG(rng_seed)
    
    # Pre-allocate subject aggregation containers
    subj_preds = Dict{Int, Vector{Int}}()
    subj_probs = Dict{Int, Vector{Float64}}()
    subj_true = Dict{Int, Int}()
    
    # Standardize
    X_train_S, X_test_S = standardize_pair(X_train, X_test)
    
    # Balance training data, strategy-based resampling
     X_train_S_bal, y_train_bal = resample_training_data(
        X_train_S, y_train;
        
        rng_seed=rng_seed
    )
    
    # Train AdaBoost
    model = AdaBoostModel(n_iter=50)
    mach = machine(model, MLJ.table(X_train_S_bal), categorical(y_train_bal))
    MLJ.fit!(mach, verbosity=0)
    
    # Predict
    yprob = MLJ.predict(mach, MLJ.table(X_test_S))
    p_pos = [Float64(pdf(s, 1)) for s in yprob]
    yhat = [p >= 0.5 ? 1 : 0 for p in p_pos]
    
    # Subject aggregation
    @inbounds for (i, subj_id) in enumerate(test_subjects)
        push!(get!(()->Int[], subj_preds, subj_id), yhat[i])
        push!(get!(()->Float64[], subj_probs, subj_id), p_pos[i])
        subj_true[subj_id] = y_test[i]
    end
    
    # Subject-level majority vote
    unique_subjects = sort(collect(keys(subj_true)))
    n_subjects = length(unique_subjects)
    
    subject_level_true = Vector{Int}(undef, n_subjects)
    subject_level_pred = Vector{Int}(undef, n_subjects)
    subject_level_prob = Vector{Float64}(undef, n_subjects)
    
    @inbounds for (idx, subj_id) in enumerate(unique_subjects)
        subject_level_true[idx] = subj_true[subj_id]
        subject_level_pred[idx] = majority_vote(subj_preds[subj_id])
        subject_level_prob[idx] = mean(subj_probs[subj_id])
    end
    
    # Calculate metrics
    mcc, f1, sen, spec, bacc, acc = calc_perf_eval_measures(
        subject_level_true, subject_level_pred)
    
    return (mcc=mcc, f1=f1, sen=sen, spec=spec, bacc=bacc, acc=acc,
            y_true=subject_level_true, y_pred=subject_level_pred,
            y_prob=subject_level_prob)
end

# ================================================================================
# Evaluate SVM-RBF
#   Evaluate SVM with RBF kernel with optional subject-level oversampling
# ================================================================================
function evaluate_svm_rbf(X_train::AbstractMatrix{<:Real}, y_train::Vector{Int},
    X_test::AbstractMatrix{<:Real}, y_test::Vector{Int}, test_subjects::Vector{Int};
    train_subjects::Vector{Int}=collect(1:length(y_train)), rng_seed::Int=42,
)::EvaluationResult
    
    # Create RNG
    rng = StableRNG(rng_seed)
    
    # Pre-allocate subject aggregation containers
    subj_preds = Dict{Int, Vector{Int}}()
    subj_probs = Dict{Int, Vector{Float64}}()
    subj_true = Dict{Int, Int}()
    
    # Standardize
    X_train_S, X_test_S = standardize_pair(X_train, X_test)
    
    # Balance training data, strategy-based resampling
    X_train_S_bal, y_train_bal = resample_training_data(
        X_train_S, y_train;
        
        rng_seed=rng_seed
    )
    
    # Build SVM model with RBF kernel
    model = SVM(kernel = LIBSVM.Kernel.RadialBasis,
                gamma  = 0.02,
                cost   = 20.0)
    
    # LIBSVM requires string categorical levels
    y_train_bal_cat = categorical(string.(y_train_bal), levels=["0","1"])
    mach = MLJ.machine(model, MLJ.table(X_train_S_bal), y_train_bal_cat)
    MLJ.fit!(mach, verbosity=0)
    
    # Predict - use string label "1" to match categorical levels
    yprob = MLJ.predict(mach, MLJ.table(X_test_S))
    p_pos = [Float64(pdf(s, "1")) for s in yprob]
    yhat = [p >= 0.5 ? 1 : 0 for p in p_pos]
    
    # Subject aggregation
    @inbounds for (i, subj_id) in enumerate(test_subjects)
        push!(get!(()->Int[], subj_preds, subj_id), yhat[i])
        push!(get!(()->Float64[], subj_probs, subj_id), p_pos[i])
        subj_true[subj_id] = y_test[i]
    end
    
    # Subject-level majority vote
    unique_subjects = sort(collect(keys(subj_true)))
    n_subjects = length(unique_subjects)
    
    subject_level_true = Vector{Int}(undef, n_subjects)
    subject_level_pred = Vector{Int}(undef, n_subjects)
    subject_level_prob = Vector{Float64}(undef, n_subjects)
    
    @inbounds for (idx, subj_id) in enumerate(unique_subjects)
        subject_level_true[idx] = subj_true[subj_id]
        subject_level_pred[idx] = majority_vote(subj_preds[subj_id])
        subject_level_prob[idx] = mean(subj_probs[subj_id])
    end
    
    # Calculate metrics
    mcc, f1, sen, spec, bacc, acc = calc_perf_eval_measures(
        subject_level_true, subject_level_pred)
    
    return (mcc=mcc, f1=f1, sen=sen, spec=spec, bacc=bacc, acc=acc,
            y_true=subject_level_true, y_pred=subject_level_pred,
            y_prob=subject_level_prob)
end

# ============================================================================
# TUNED CLASSIFIER EVALUATION FUNCTIONS  
# ============================================================================

# =====================================================================================
# Predict with Tuned SVM
#   Helper function for tuned SVM with hyperparameter search.
# =====================================================================================
function predict_with_tuned_svm( Xtr::AbstractMatrix, ytr::Vector{Int}, Xte::AbstractMatrix;
    train_subjects::Vector{Int}, include_standardizer::Bool = false, n_inner_folds::Int = 3, n_samples::Int = 50, res::Int = 50,
    rng::Int = 42, verbosity::Int = 0)
    # Pipeline 
    pipe = include_standardizer ?
        Pipeline(stand = Standardizer(),
                 svm   = SVM(kernel = LIBSVM.Kernel.RadialBasis);
                 prediction_type = :probabilistic) :
        Pipeline(svm = SVM(kernel = LIBSVM.Kernel.RadialBasis);
                 prediction_type = :probabilistic)

    # Ranges
    r_cost  = range(pipe, :(svm.cost),  lower = 1e-2,  upper = 1e2,  scale = :log10)
    r_gamma = range(pipe, :(svm.gamma), lower = 1e-3, upper = 1.0, scale = :log10)
    ranges  = [r_cost, r_gamma]

    # LIBSVM requires string categorical targets
    ytr_cat = categorical(string.(ytr), levels=["0","1"])

    # Leakage-free inner folds (subject-level CV)
    inner_folds_named = stratified_subject_level_cv(
        train_subjects,
        ytr,
        n_inner_folds;
        random_seed = rng,
    )

    # MLJ expects Vector{Tuple{train_rows, test_rows}}
    inner_folds = [(f.train, f.test) for f in inner_folds_named]

    tuner = TunedModel(
        model      = pipe,
        tuning     = Grid(resolution= res),
        resampling = inner_folds,
        ranges     = ranges,
        measure    = MatthewsCorrelation(),
        n          = n_samples,
        train_best = true,
    )
        
    mach = machine(tuner, MLJ.table(Xtr), ytr_cat)
    MLJ.fit!(mach, verbosity = verbosity)

    yprob       = MLJ.predict(mach, MLJ.table(Xte))
    best_model  = fitted_params(mach).best_model
    best_report = report(mach)

    return yprob, best_model, best_report
end
# =====================================================================================
# Evaluate tuned SVM
#   with hyperparameter optimization.
# =====================================================================================
function evaluate_svm_rbf_tuned(
    X_train::AbstractMatrix{<:Real}, y_train::Vector{Int}, X_test::AbstractMatrix{<:Real},  y_test::Vector{Int},
    test_subjects::Vector{Int}; train_subjects::Vector{Int}, rng_seed::Int = 42, reso::Int = 50 )::TunedEvaluationResult
    
    # Create RNG
    rng = StableRNG(rng_seed)
    
    # Pre-allocate subject aggregation containers
    subj_preds = Dict{Int, Vector{Int}}()
    subj_probs = Dict{Int, Vector{Float64}}()
    subj_true = Dict{Int, Int}()
    
    # Standardize
    X_train_S, X_test_S = standardize_pair(X_train, X_test)
    
    # Balance training data, strategy-based resampling
    X_train_S_bal, y_train_bal = resample_training_data(
        X_train_S, y_train;
        
        rng_seed=rng_seed
    )
    
	# Note: For tuned models, we still need train_subjects for inner CV
    # After resampling, we can't track which subject each row belongs to
    # So we pass the balanced data but use original train_subjects structure for inner CV
    # This is acceptable because inner CV re-splits the balanced training data
    n_resampled = length(y_train_bal)
    train_subjects_bal = collect(1:n_resampled)  # Each row becomes its own "subject"
	
	
    # Tune on inner folds and predict
    yprob, best_model, best_report = predict_with_tuned_svm(
        X_train_S_bal, y_train_bal, X_test_S;
        train_subjects = train_subjects_bal,
        n_inner_folds  = 3,
        n_samples      = 50,
        res           = reso,
        rng            = rng_seed,        
        verbosity      = 0,
    )
    
    # Extract probabilities and predictions
    # Note: yprob has string levels ("0", "1") to match LIBSVM requirements
    p_pos = [Float64(pdf(s, "1")) for s in yprob]
    yhat = [p >= 0.5 ? 1 : 0 for p in p_pos]
    
    # Subject aggregation
    @inbounds for (i, subj_id) in enumerate(test_subjects)
        push!(get!(()->Int[], subj_preds, subj_id), yhat[i])
        push!(get!(()->Float64[], subj_probs, subj_id), p_pos[i])
        subj_true[subj_id] = y_test[i]
    end
    
    # Subject-level majority vote
    unique_subjects = sort(collect(keys(subj_true)))
    n_subjects = length(unique_subjects)
    
    subject_level_true = Vector{Int}(undef, n_subjects)
    subject_level_pred = Vector{Int}(undef, n_subjects)
    subject_level_prob = Vector{Float64}(undef, n_subjects)
    
    @inbounds for (idx, subj_id) in enumerate(unique_subjects)
        subject_level_true[idx] = subj_true[subj_id]
        subject_level_pred[idx] = majority_vote(subj_preds[subj_id])
        subject_level_prob[idx] = mean(subj_probs[subj_id])
    end
    
    # Calculate metrics
    mcc, f1, sen, spec, bacc, acc = calc_perf_eval_measures(
        subject_level_true, subject_level_pred)
    
    return (mcc=mcc, f1=f1, sen=sen, spec=spec, bacc=bacc, acc=acc,
            y_true=subject_level_true, y_pred=subject_level_pred,
            y_prob=subject_level_prob,
            best_model=best_model)
end


# ================================================================================
# Function to use tuned RF for prediction
#       Helper function for tuned Random Forest with hyperparameter search.
# ================================================================================
function predict_with_tuned_random_forest(Xtr::AbstractMatrix, ytr::Vector{Int}, Xte::AbstractMatrix;
    train_subjects::Vector{Int}, n_inner_folds::Int = 3, n_samples::Int = 30, res::Int = 30, rng::Int = 42, verbosity::Int = 0)
    
    # Create RandomForest model with ranges
    rf_model = MLJDecisionTreeInterface.RandomForestClassifier(rng=rng)
    
    # Define hyperparameter ranges
    r_n_trees = range(rf_model, :n_trees, lower=50, upper=200)
    r_max_depth = range(rf_model, :max_depth, lower=5, upper=20)
    ranges = [r_n_trees, r_max_depth]
    
    # Categorical targets
    ytr_cat = categorical(ytr)
    
    # Leakage-free inner folds (subject-level CV)
    inner_folds_named = stratified_subject_level_cv(
        train_subjects,
        ytr,
        n_inner_folds;
        random_seed = rng,
    )
    
    # MLJ expects Vector{Tuple{train_rows, test_rows}}
    inner_folds = [(f.train, f.test) for f in inner_folds_named]
    
    tuner = TunedModel(
        model      = rf_model,
        tuning     = Grid(resolution=res),
        resampling = inner_folds,
        ranges     = ranges,
        measure    = MatthewsCorrelation(),
        n          = n_samples,
        train_best = true,
    )
    
    mach = machine(tuner, MLJ.table(Xtr), ytr_cat)
    MLJ.fit!(mach, verbosity = verbosity)
    
    yprob       = MLJ.predict(mach, MLJ.table(Xte))
    best_model  = fitted_params(mach).best_model
    best_report = report(mach)
    
    return yprob, best_model, best_report
end


# ================================================================================
# function to evaluate Tuned RF
#           Evaluate tuned Random Forest with hyperparameter optimization.
# ================================================================================
function evaluate_random_forests_tuned(X_train::AbstractMatrix{<:Real}, y_train::Vector{Int},
    X_test::AbstractMatrix{<:Real},  y_test::Vector{Int}, test_subjects::Vector{Int}; train_subjects::Vector{Int},
    rng_seed::Int = 42, reso::Int = 30)::TunedEvaluationResult
    
    # Create RNG
    rng = StableRNG(rng_seed)
    
    # Pre-allocate subject aggregation containers
    subj_preds = Dict{Int, Vector{Int}}()
    subj_probs = Dict{Int, Vector{Float64}}()
    subj_true = Dict{Int, Int}()
    
    # Standardize
    X_train_S, X_test_S = standardize_pair(X_train, X_test)
    
    # Balance training data, strategy-based resampling
    X_train_S_bal, y_train_bal = resample_training_data(
        X_train_S, y_train;
        
        rng_seed=rng_seed
    )
    
	# Note: For tuned models, we still need train_subjects for inner CV
    # After resampling, we can't track which subject each row belongs to
    # So we pass the balanced data but use original train_subjects structure for inner CV
    # This is acceptable because inner CV re-splits the balanced training data
    n_resampled = length(y_train_bal)
    train_subjects_bal = collect(1:n_resampled)  # Each row becomes its own "subject"
	
    # Tune and predict
    yprob, best_model, best_report = predict_with_tuned_random_forest(
        X_train_S_bal, y_train_bal, X_test_S;
        train_subjects = train_subjects_bal,
        n_inner_folds  = 3,
        n_samples      = 30,
        res            = reso,
        rng            = rng_seed,
        verbosity      = 0,
    )
    
    # Extract probabilities and predictions
    p_pos = [Float64(pdf(s, 1)) for s in yprob]
    yhat = [p >= 0.5 ? 1 : 0 for p in p_pos]
    
    # Subject aggregation
    @inbounds for (i, subj_id) in enumerate(test_subjects)
        push!(get!(()->Int[], subj_preds, subj_id), yhat[i])
        push!(get!(()->Float64[], subj_probs, subj_id), p_pos[i])
        subj_true[subj_id] = y_test[i]
    end
    
    # Subject-level majority vote
    unique_subjects = sort(collect(keys(subj_true)))
    n_subjects = length(unique_subjects)
    
    subject_level_true = Vector{Int}(undef, n_subjects)
    subject_level_pred = Vector{Int}(undef, n_subjects)
    subject_level_prob = Vector{Float64}(undef, n_subjects)
    
    @inbounds for (idx, subj_id) in enumerate(unique_subjects)
        subject_level_true[idx] = subj_true[subj_id]
        subject_level_pred[idx] = majority_vote(subj_preds[subj_id])
        subject_level_prob[idx] = mean(subj_probs[subj_id])
    end
    
    # Calculate metrics
    mcc, f1, sen, spec, bacc, acc = calc_perf_eval_measures(
        subject_level_true, subject_level_pred)
    
    return (mcc=mcc, f1=f1, sen=sen, spec=spec, bacc=bacc, acc=acc,
            y_true=subject_level_true, y_pred=subject_level_pred,
            y_prob=subject_level_prob,
            best_model=best_model)
end


# ================================================================================
# Function to use tuned AdaBoost for prediction
#           function for tuned AdaBoost with hyperparameter search.
# ================================================================================
function predict_with_tuned_adaboost(Xtr::AbstractMatrix, ytr::Vector{Int}, Xte::AbstractMatrix;
    train_subjects::Vector{Int}, n_inner_folds::Int = 3, n_samples::Int = 30, res::Int = 30,rng::Int = 47,
    verbosity::Int = 0)
    # Create AdaBoost model
    ada_model = AdaBoostModel()
    
    # Define hyperparameter ranges
    r_n_iter = range(ada_model, :n_iter, lower=20, upper=100)
    ranges = [r_n_iter]
    
    # Categorical targets
    ytr_cat = categorical(ytr)
    
    # Leakage-free inner folds (subject-level CV)
    inner_folds_named = stratified_subject_level_cv(
        train_subjects,
        ytr,
        n_inner_folds;
        random_seed = rng,
    )
    
    # MLJ expects Vector{Tuple{train_rows, test_rows}}
    inner_folds = [(f.train, f.test) for f in inner_folds_named]
    
    tuner = TunedModel(
        model      = ada_model,
        tuning     = Grid(resolution=res),
        resampling = inner_folds,
        ranges     = ranges,
        measure    = MatthewsCorrelation(),
        n          = n_samples,
        train_best = true,
    )
    
    mach = machine(tuner, MLJ.table(Xtr), ytr_cat)
    MLJ.fit!(mach, verbosity = verbosity)
    
    yprob       = MLJ.predict(mach, MLJ.table(Xte))
    best_model  = fitted_params(mach).best_model
    best_report = report(mach)
    
    return yprob, best_model, best_report
end

# ================================================================================
#function to evaluate Tuned AdaBoost
# ================================================================================
function evaluate_adaboost_tuned(X_train::AbstractMatrix{<:Real}, y_train::Vector{Int},
    X_test::AbstractMatrix{<:Real},  y_test::Vector{Int}, test_subjects::Vector{Int};
    train_subjects::Vector{Int}, rng_seed::Int = 42, reso::Int = 30)::TunedEvaluationResult
    
    # Create RNG
    rng = StableRNG(rng_seed)
    
    # Pre-allocate subject aggregation containers
    subj_preds = Dict{Int, Vector{Int}}()
    subj_probs = Dict{Int, Vector{Float64}}()
    subj_true = Dict{Int, Int}()
    
    # Standardize
    X_train_S, X_test_S = standardize_pair(X_train, X_test)
    
    # Balance training data, strategy-based resampling
    X_train_S_bal, y_train_bal = resample_training_data(
        X_train_S, y_train;
        
        rng_seed=rng_seed
    )
    
	# Note: For tuned models, we still need train_subjects for inner CV
    # After resampling, we can't track which subject each row belongs to
    # So we pass the balanced data but use original train_subjects structure for inner CV
    # This is acceptable because inner CV re-splits the balanced training data
    n_resampled = length(y_train_bal)
    train_subjects_bal = collect(1:n_resampled)  # Each row becomes its own "subject"
	
    # Tune and predict
    yprob, best_model, best_report = predict_with_tuned_adaboost(
        X_train_S_bal, y_train_bal, X_test_S;
        train_subjects = train_subjects_bal,
        n_inner_folds  = 3,
        n_samples      = 30,
        res            = reso,
        rng            = rng_seed,
        verbosity      = 0,
    )
    
    # Extract probabilities and predictions
    p_pos = [Float64(pdf(s, 1)) for s in yprob]
    yhat = [p >= 0.5 ? 1 : 0 for p in p_pos]
    
    # Subject aggregation
    @inbounds for (i, subj_id) in enumerate(test_subjects)
        push!(get!(()->Int[], subj_preds, subj_id), yhat[i])
        push!(get!(()->Float64[], subj_probs, subj_id), p_pos[i])
        subj_true[subj_id] = y_test[i]
    end
    
    # Subject-level majority vote
    unique_subjects = sort(collect(keys(subj_true)))
    n_subjects = length(unique_subjects)
    
    subject_level_true = Vector{Int}(undef, n_subjects)
    subject_level_pred = Vector{Int}(undef, n_subjects)
    subject_level_prob = Vector{Float64}(undef, n_subjects)
    
    @inbounds for (idx, subj_id) in enumerate(unique_subjects)
        subject_level_true[idx] = subj_true[subj_id]
        subject_level_pred[idx] = majority_vote(subj_preds[subj_id])
        subject_level_prob[idx] = mean(subj_probs[subj_id])
    end
    
    # Calculate metrics
    mcc, f1, sen, spec, bacc, acc = calc_perf_eval_measures(
        subject_level_true, subject_level_pred)
    
    return (mcc=mcc, f1=f1, sen=sen, spec=spec, bacc=bacc, acc=acc,
            y_true=subject_level_true, y_pred=subject_level_pred,
            y_prob=subject_level_prob,
            best_model=best_model)
end

# ================================================================================
# function to use Tuned NN for prediction
#                                   with hyperparameter search.
# ================================================================================
function predict_with_tuned_neural(Xtr::AbstractMatrix, ytr::Vector{Int}, Xte::AbstractMatrix;
    train_subjects::Vector{Int}, n_inner_folds::Int = 3, n_samples::Int = 30, res::Int = 30, rng::Int = 42,
    verbosity::Int = 0)
    # Create Neural Network model
    n_features = size(Xtr, 2)
    nn_model = nn_builder(n_features)
    
    # Define hyperparameter ranges  
    r_lambda = range(nn_model, :lambda, lower=0.001, upper=0.03, scale=:log10)
    r_alpha = range(nn_model, :alpha, lower=0.0, upper=0.3)
    #Just tune lambda (L2) ,as L1 is good for feature selection, butalready doing feature selection separately.
    ranges = [r_lambda, r_alpha]
    #ranges = [r_lambda]
    
    # Categorical targets
    ytr_cat = categorical(ytr)
    
    # Leakage-free inner folds (subject-level CV)
    inner_folds_named = stratified_subject_level_cv(
        train_subjects,
        ytr,
        n_inner_folds;
        random_seed = rng,
    )
    
    # MLJ expects Vector{Tuple{train_rows, test_rows}}
    inner_folds = [(f.train, f.test) for f in inner_folds_named]
    
    tuner = TunedModel(
        model      = nn_model,
        tuning     = Grid(resolution=res),
        resampling = inner_folds,
        ranges     = ranges,
        measure    = MatthewsCorrelation(),
        n          = n_samples,
        train_best = true,
    )
    
    mach = machine(tuner, MLJ.table(Xtr), ytr_cat)
    MLJ.fit!(mach, verbosity = verbosity)
    
    yprob       = MLJ.predict(mach, MLJ.table(Xte))
    best_model  = fitted_params(mach).best_model
    best_report = report(mach)
    training_losses = if hasproperty(best_report, :model_report) && !isnothing(best_report.model_report)
        best_report.model_report.training_losses
    else
        # MLJTuning does not expose the best model's sub-report; refit to capture
        # losses. Snapshot and restore the task-local RNG so this extra fit never
        # perturbs the trajectory seen by the rest of the fold.
        rng_state = copy(Random.default_rng())
        Random.seed!(rng)
        bm_mach = machine(best_model, MLJ.table(Xtr), ytr_cat)
        MLJ.fit!(bm_mach, verbosity=0)
        _losses = report(bm_mach).training_losses
        copy!(Random.default_rng(), rng_state)
        _losses
    end

    return yprob, best_model, best_report, training_losses
end

# ================================================================================
# Neural Network tuned
#               Evaluate tuned Neural Network with hyperparameter optimization
# ================================================================================
function evaluate_neural_network_tuned( X_train::AbstractMatrix{<:Real}, y_train::Vector{Int},
    X_test::AbstractMatrix{<:Real},  y_test::Vector{Int}, test_subjects::Vector{Int};
    train_subjects::Vector{Int}, rng_seed::Int = 42, reso::Int =30 )
    
    # Create RNG
    rng = StableRNG(rng_seed)
    
    # Pre-allocate subject aggregation containers
    subj_preds = Dict{Int, Vector{Int}}()
    subj_probs = Dict{Int, Vector{Float64}}()
    subj_true = Dict{Int, Int}()
    
    # Standardize
    X_train_S, X_test_S = standardize_pair(X_train, X_test)
    
    # Balance training data, strategy-based resampling
    X_train_S_bal, y_train_bal = resample_training_data(
        X_train_S, y_train;
        
        rng_seed=rng_seed
    )
    
	# Note: For tuned models, we still need train_subjects for inner CV
    # After resampling, we can't track which subject each row belongs to
    # So we pass the balanced data but use original train_subjects structure for inner CV
    # This is acceptable because inner CV re-splits the balanced training data
    n_resampled = length(y_train_bal)
    train_subjects_bal = collect(1:n_resampled)  # Each row becomes its own "subject"
	
    # Tune and predict
    yprob, best_model, best_report, training_losses = predict_with_tuned_neural(
        X_train_S_bal, y_train_bal, X_test_S;
        train_subjects = train_subjects_bal,
        n_inner_folds  = 3,
        n_samples      = 3,
        res            = reso, 
        rng            = rng_seed,
        verbosity      = 0,
    )
    
    # Extract probabilities and predictions
    p_pos = [Float64(pdf(s, 1)) for s in yprob]
    yhat = [p >= 0.5 ? 1 : 0 for p in p_pos]
    
    # Subject aggregation
    @inbounds for (i, subj_id) in enumerate(test_subjects)
        push!(get!(()->Int[], subj_preds, subj_id), yhat[i])
        push!(get!(()->Float64[], subj_probs, subj_id), p_pos[i])
        subj_true[subj_id] = y_test[i]
    end
    
    # Subject-level majority vote
    unique_subjects = sort(collect(keys(subj_true)))
    n_subjects = length(unique_subjects)
    
    subject_level_true = Vector{Int}(undef, n_subjects)
    subject_level_pred = Vector{Int}(undef, n_subjects)
    subject_level_prob = Vector{Float64}(undef, n_subjects)
    
    @inbounds for (idx, subj_id) in enumerate(unique_subjects)
        subject_level_true[idx] = subj_true[subj_id]
        subject_level_pred[idx] = majority_vote(subj_preds[subj_id])
        subject_level_prob[idx] = mean(subj_probs[subj_id])
    end
    
    # Calculate metrics
    mcc, f1, sen, spec, bacc, acc = calc_perf_eval_measures(
        subject_level_true, subject_level_pred)

    result = (mcc=mcc, f1=f1, sen=sen, spec=spec, bacc=bacc, acc=acc,
              y_true=subject_level_true, y_pred=subject_level_pred,
              y_prob=subject_level_prob,
              best_model=best_model)
    return result, training_losses
end

# ====================================================================
# Feature Selection Summary - All Datasets and Methods
# arg Method
# ====================================================================
function feat_sel_sum(method)    
    # Map dataset names to their X, y, and feature_names
    dataset_order = ["baseline", "sex aware", "male", "female"]
	dataset_map = Dict(
        "baseline" => (Xb, yb, feature_names_b),
        "sex aware" => (Xs, yb, feature_names_s),
        "male" => (Xm, ym, feature_names_m),
        "female" => (Xf, yf, feature_names_f)
    )
    
    for dataset_name in dataset_order
        println("\n" * "="^70)
        println("DATASET: $(uppercase(dataset_name))")
        println("="^70 * "\n")
        
        X, y, feature_names = dataset_map[dataset_name]
        nested_cv_results = all_results[dataset_name]
        
        nested_cv_feature_selection_summary(
			nested_cv_results,
			X,
			y,
			feature_names,
			method,
			top_n_feats_candidates
		)
        
    end
end

# ====================================================================
# Feature categor distribution - All Datasets and Methods
# 			Arg: Method to be used to generate table
# ====================================================================
function feat_cat_dist_allData(method)
    dataset_order = ["baseline", "sex aware", "male", "female"]
        
    # Map dataset names to their feature_names
    feature_names_map = Dict(
        "baseline" => feature_names_b,
        "sex aware" => feature_names_s,
        "male" => feature_names_m,
        "female" => feature_names_f
    )
    
    for dataset_name in dataset_order
        println("\n" * "="^70)
        println("DATASET: $(uppercase(dataset_name))")
        println("="^70 * "\n")
        
        feature_names = feature_names_map[dataset_name]
        nested_cv_results = all_results[dataset_name]
        
		nested_cv_feature_category_distribution(
			nested_cv_results,
			feature_names,
			method,
			top_n_feats_candidates
		)
        
    end  
end
# ============================================================================
# Export MCC Comparison and Tuned Parameters to CSV
# Args:
#  all_results: Dictionary containing nested CV results for all datasets
#  
#  Format: {"baseline" => nested_cv_results, "sex aware" => ..., etc.}
#  
#  top_n_feats_candidates: Vector of feat counts tested 
#                                                (e.g., [10, 20, 30, ..., 100])
#  
#  output_dir: (Optional) Directory path for output files.
#                                               Default: "." (current directory)
#   
#  file_prefix: (Optional) Prefix to add to all filenames. Default: "" (no prefix)
# 
# Exports:
#        Individual files per dataset (8 files):
#            - mcc_comparison_<dataset>.csv
#            - best_tuned_parameters_<dataset>.csv
#        Combined files across all datasets (2 files):
#            - mcc_comparison_all_datasets.csv
#            - best_tuned_parameters_all_datasets.csv
# 
# Returns:
#        Named tuple with: mcc_rows, tuned_rows, files_created, output_dir
# ============================================================================
function export_all_results_to_csv(all_results::Dict, top_n_feats_candidates::Vector{Int}, strategy::String; output_dir::String = ".")
    
    println("\n" * "="^60)
    println("Exporting results to CSV files...")
    println("="^60 * "\n")
    
    # Define classifiers
    classifiers = [
        "RandomForest", "RandomForest_TUNED",
        "NeuralNetwork", "NeuralNetwork_TUNED",
        "NaiveBayes",
        "LogisticRegression",
        "AdaBoost", "AdaBoost_TUNED",
        "SVM_RBF", "SVM_RBF_TUNED"
    ]
    
    methods = ["mRMR", "Boruta"]
    
    # Export for each dataset
    for (dataset_name, nested_cv_results) in all_results
        
        println("Exporting $dataset_name...")
        
        # Initialize results DataFrame
        results_df = DataFrame(
            Dataset = String[],
            Method = String[],
            N_Features = Int[],
            Classifier = String[],
            Fold = Int[],
            MCC = Float64[],
            F1 = Float64[],
            Sensitivity = Float64[],
            Specificity = Float64[],
            Balanced_Accuracy = Float64[],
            Accuracy = Float64[]
        )
        
        # Collect results from all outer folds
        n_outer_folds = length(nested_cv_results)
        
        for outer_fold_idx in 1:n_outer_folds
            for method in methods
                for n_features in top_n_feats_candidates
                    feature_count = "$(n_features)_features"
                    
                    for classifier in classifiers
                        # Get results for this combination
                        if haskey(nested_cv_results[outer_fold_idx][method], feature_count) &&
                           haskey(nested_cv_results[outer_fold_idx][method][feature_count], classifier)
                            
                            metrics = nested_cv_results[outer_fold_idx][method][feature_count][classifier]
                            
                            push!(results_df, (
                                Dataset = dataset_name,
                                Method = method,
                                N_Features = n_features,
                                Classifier = classifier,
                                Fold = outer_fold_idx,
                                MCC = metrics.mcc,
                                F1 = metrics.f1,
                                Sensitivity = metrics.sen,
                                Specificity = metrics.spec,
                                Balanced_Accuracy = metrics.bacc,
                                Accuracy = metrics.acc
                            ))
                        end
                    end
                end
            end
        end
        
        # Save to CSV
		date= (Dates.format(now(), "yyyymmdd_HHMMSS"))
        filename = joinpath(output_dir, "results_$(dataset_name)_($strategy)_$date.csv")
        CSV.write(filename, results_df)
        println("  ✓ Saved: $filename ($(nrow(results_df)) rows)")
    end
    
    println("\n" * "="^60)
    println("Export completed successfully!")
    println("="^60 * "\n")
end

# ============================================================================
# Export Cochran's Q Test and McNemar p-values Tables to CSV
# Args:
#        all_results: Dictionary containing nested CV results for all datasets
#        top_n_feats_candidates: Vector of feature counts tested
#        output_dir: (Optional) Directory path for output files. Default: "." 
#        file_prefix: (Optional) Prefix to add to all filenames. Default: ""
# 
# Exports:
#        For each dataset (includes ALL feature counts):
#            - cochrans_q_<dataset>_all_features.csv
#            - mcnemar_<dataset>_all_features.csv
#        Combined files (all datasets, all feature counts):
#            - cochrans_q_all_datasets_all_features.csv
#            - mcnemar_all_datasets_all_features.csv
# ============================================================================
function export_statistical_tests_to_csv(all_results, top_n_feats_candidates; 
                                         output_dir::String = ".",
                                         file_prefix::String = "")
    
    dataset_order = ["baseline", "sex aware", "male", "female"]
    
    # Storage for combined DataFrames
    all_cochran_dfs = []
    all_mcnemar_dfs = []
    
    println("Exporting statistical test results for all feature counts...")
    println("="^70)
    println("Feature counts: $(sort(top_n_feats_candidates))")
    println("="^70)
    
    for dataset_name in dataset_order
        nested_cv_results = all_results[dataset_name]
        file_suffix = replace(dataset_name, " " => "_")
        
        # Storage for this dataset across all feature counts
        dataset_cochran_dfs = []
        dataset_mcnemar_dfs = []
        
        # Loop through all feature counts
        for n_features in sort(top_n_feats_candidates)
            # Generate Cochran's Q DataFrame
            cochran_df = generate_cochrans_q_df(nested_cv_results, n_features)
            cochran_df[!, :n_features] .= n_features
            cochran_df[!, :dataset] .= dataset_name
            push!(dataset_cochran_dfs, cochran_df)
            
            # Generate McNemar DataFrame
            mcnemar_df = generate_mcnemar_df(nested_cv_results, n_features)
            mcnemar_df[!, :n_features] .= n_features
            mcnemar_df[!, :dataset] .= dataset_name
            push!(dataset_mcnemar_dfs, mcnemar_df)
        end
        
        # Combine all feature counts for this dataset
        dataset_cochran_combined = vcat(dataset_cochran_dfs...)
        dataset_mcnemar_combined = vcat(dataset_mcnemar_dfs...)
        
        # Reorder columns: n_features first (without dataset for individual files)
        select!(dataset_cochran_combined, :n_features, Not([:n_features, :dataset]), :dataset)
        select!(dataset_mcnemar_combined, :n_features, Not([:n_features, :dataset]), :dataset)
        
        # Construct filenames
        cochran_filename = isempty(file_prefix) ? 
            "cochrans_q_$(file_suffix)_all_features.csv" :
            "$(file_prefix)_cochrans_q_$(file_suffix)_all_features.csv"
        
        mcnemar_filename = isempty(file_prefix) ?
            "mcnemar_$(file_suffix)_all_features.csv" :
            "$(file_prefix)_mcnemar_$(file_suffix)_all_features.csv"
        
        # Full paths
        cochran_path = joinpath(output_dir, cochran_filename)
        mcnemar_path = joinpath(output_dir, mcnemar_filename)
        
        # Export individual files (without dataset column)
        CSV.write(cochran_path, dataset_cochran_combined[:, Not(:dataset)])
        CSV.write(mcnemar_path, dataset_mcnemar_combined[:, Not(:dataset)])
        
        println("✓ $(dataset_name): $(length(top_n_feats_candidates)) feature counts")
        println("  - $(cochran_filename) ($(nrow(dataset_cochran_combined)) rows)")
        println("  - $(mcnemar_filename) ($(nrow(dataset_mcnemar_combined)) rows)")
        
        # Store for combined export
        push!(all_cochran_dfs, dataset_cochran_combined)
        push!(all_mcnemar_dfs, dataset_mcnemar_combined)
    end
    
    println("\n" * "="^70)
    println("Exporting combined files...")
    println("="^70)
    
    # Combine all datasets
    combined_cochran = vcat(all_cochran_dfs...)
    combined_mcnemar = vcat(all_mcnemar_dfs...)
    
    # Reorder columns: dataset first, then n_features
    select!(combined_cochran, :dataset, :n_features, Not([:dataset, :n_features]))
    select!(combined_mcnemar, :dataset, :n_features, Not([:dataset, :n_features]))
    
    # Construct combined filenames
    combined_cochran_filename = isempty(file_prefix) ?
        "cochrans_q_all_datasets_all_features.csv" :
        "$(file_prefix)_cochrans_q_all_datasets_all_features.csv"
    
    combined_mcnemar_filename = isempty(file_prefix) ?
        "mcnemar_all_datasets_all_features.csv" :
        "$(file_prefix)_mcnemar_all_datasets_all_features.csv"
    
    # Full paths for combined files
    combined_cochran_path = joinpath(output_dir, combined_cochran_filename)
    combined_mcnemar_path = joinpath(output_dir, combined_mcnemar_filename)
    
    # Export combined files
    CSV.write(combined_cochran_path, combined_cochran)
    CSV.write(combined_mcnemar_path, combined_mcnemar)
    
    println("✓ $(combined_cochran_filename) ($(nrow(combined_cochran)) rows)")
    println("✓ $(combined_mcnemar_filename) ($(nrow(combined_mcnemar)) rows)")
    println("\n✓✓✓ Statistical test exports completed successfully! ✓✓✓")
    println("\nSummary:")
    println("  - Feature counts tested: $(length(top_n_feats_candidates))")
    println("  - Datasets: $(length(dataset_order))")
    println("  - Total rows per combined file: $(nrow(combined_cochran))")
    
    return (
        cochran_rows = nrow(combined_cochran),
        mcnemar_rows = nrow(combined_mcnemar),
        files_created = 8 + 2,  # 4 datasets × 2 tests + 2 combined files
        n_feature_counts = length(top_n_feats_candidates),
        output_dir = output_dir
    )
end
# ============================================================================
# Generate Cochran's Q DataFrame (without printing)
# ============================================================================
function generate_cochrans_q_df(nested_cv_results, n_features_selected)
    classifiers = [
        "RandomForest", "RandomForest_TUNED",
        "NeuralNetwork", "NeuralNetwork_TUNED",
        "NaiveBayes",
        "LogisticRegression",
        "AdaBoost", "AdaBoost_TUNED",
        "SVM_RBF", "SVM_RBF_TUNED"]
    classifier_display_names = [
        "RF", "RF (Tuned)",
        "NeuralNet", "NeuralNet (Tuned)",
        "NaiveBayes",
        "LogReg",
        "AdaBoost", "AdaBoost (Tuned)",
        "SVM-RBF", "SVM-RBF (Tuned)"]
    
    methods = ["mRMR", "Boruta"]
    feature_count = "$(n_features_selected)_features"
    
    Qvals = Float64[]
    pvals = Float64[]
    sigs = Bool[]
    
    for (clf_name, clf_display) in zip(classifiers, classifier_display_names)
        y_true_all = Int[]
        predictions_matrix = Vector{Int}[]
        
        for method_name in methods
            method_preds = []
            
            for outer_fold_idx in 1:length(nested_cv_results)
                metrics = nested_cv_results[outer_fold_idx][method_name][feature_count][clf_name]
                
                if method_name == "mRMR"
                    append!(y_true_all, metrics.y_true)
                end
                
                append!(method_preds, metrics.y_pred)
            end
            
            push!(predictions_matrix, method_preds)
        end
        
        Ŷ = hcat(predictions_matrix...)
        result = cochrans_q(y_true_all, Ŷ)
        
        push!(Qvals, result.Q)
        push!(pvals, result.p_value)
        push!(sigs, result.significant)
    end
    
    return DataFrame(
        Classifier = classifier_display_names,
        Q = round.(Qvals, digits=4),
        df = fill(2, length(classifiers)),
        p_value = round.(pvals, digits=4),
        significant = sigs
    )
end
# ============================================================================
# Helper: Generate McNemar DataFrame (without printing)
# ============================================================================
function generate_mcnemar_df(nested_cv_results, n_features_selected)
    classifiers = ["RandomForest","NeuralNetwork", "NaiveBayes", 
                   "LogisticRegression", "AdaBoost", "SVM_RBF"]
    clf_abbrev = ["RF", "NN", "NB", "LR", "Ada", "SVM"]
    
    methods = ["mRMR", "Boruta"]
    feature_count = "$(n_features_selected)_features"
    
    comps = [(a, b, "$a vs $b") for (a, b) in combinations(methods, 2)]
    summary_tbl = DataFrame(Comparison = [name for (_, _, name) in comps])
    
    for (clf_name, clf_short) in zip(classifiers, clf_abbrev)
        pcol = Float64[]
        
        method_predictions = Dict{String, Vector{Int}}()
        y_true_all = Int[]
        
        for method_name in methods
            method_preds = Int[]
            
            for outer_fold_idx in 1:length(nested_cv_results)
                metrics = nested_cv_results[outer_fold_idx][method_name][feature_count][clf_name]
                
                if method_name == "mRMR"
                    append!(y_true_all, metrics.y_true)
                end
                
                append!(method_preds, metrics.y_pred)
            end
            
            method_predictions[method_name] = method_preds
        end
        
        for (method_a, method_b, _) in comps
            y_pred_a = method_predictions[method_a]
            y_pred_b = method_predictions[method_b]
            
            p_value = mcnemar_test(y_true_all, y_pred_a, y_pred_b).p_value
            push!(pcol, round(p_value, digits = 4))
        end
        
        summary_tbl[!, clf_short] = pcol
    end
    
    return summary_tbl
end

# ============================================================================
# Materialize Baseline Sex Results as Virtual Datasets
#
# Create virtual dataset results for baseline_male/baseline_female from baseline fold results.
# Keeps the same nested_cv_results structure (outer_fold -> method -> feature_count -> classifiers).
# Copies selected_features_* entries unchanged.
# For each classifier, uses "<classifier>_sex" if present, and maps to the chosen sex result.
#
# Args:
#    baseline_nested_cv_results: The baseline dataset results from all_results_adasyn["baseline"]
#    sex: :male or :female
#
# Returns:
#   A Dict with the same structure as nested_cv_results, but with sex-specific metrics
# ============================================================================
function materialize_baseline_sex_results(
    baseline_nested_cv_results::Dict,
    sex::Symbol
)::Dict
    
    # Validate input
    @assert sex in [:male, :female] "sex must be :male or :female"
    
    # Initialize output dictionary
    virtual_results = Dict()
    
    # List of classifiers that may have sex splits
    classifiers = [
        "RandomForest", "RandomForest_TUNED",
        "NeuralNetwork", "NeuralNetwork_TUNED",
        "NaiveBayes",
        "LogisticRegression",
        "AdaBoost", "AdaBoost_TUNED",
        "SVM_RBF", "SVM_RBF_TUNED"
    ]
    
    # Loop through outer folds
    for outer_fold_idx in sort(collect(keys(baseline_nested_cv_results)))
        fold_dict = baseline_nested_cv_results[outer_fold_idx]
        virtual_fold_dict = Dict()
        
        # Loop through methods (mRMR, Boruta)
        for (method_name, method_dict) in fold_dict
            virtual_method_dict = Dict()
            
            # Copy each entry in the method dictionary
            for (key, value) in method_dict
                # Preserve selected_features entries unchanged
                if startswith(string(key), "selected_features")
                    virtual_method_dict[key] = value
                    continue
                end
                
                # This is a feature count entry (e.g., "10_features")
                if value isa Dict
                    virtual_feature_dict = Dict()
                    
                    # Process each classifier in this feature count
                    for classifier in classifiers
                        # Check if the sex-split version exists
                        sex_key = "$(classifier)_sex"
                        if haskey(value, sex_key)
                            sex_split = value[sex_key]
                            # Extract the appropriate sex result
                            if sex == :male && haskey(sex_split, :male)
                                virtual_feature_dict[classifier] = sex_split.male
                            elseif sex == :female && haskey(sex_split, :female)
                                virtual_feature_dict[classifier] = sex_split.female
                            end
                        end
                    end
                    
                    virtual_method_dict[key] = virtual_feature_dict
                end
            end
            
            virtual_fold_dict[method_name] = virtual_method_dict
        end
        
        virtual_results[outer_fold_idx] = virtual_fold_dict
    end
    
    return virtual_results
end

# ============================================================================
# Materialize Baseline Sex SHAP as Virtual Datasets
# ============================================================================
# Create virtual SHAP dicts for baseline_male/baseline_female without recomputation.
# Input is all_datasets_shap["baseline"] which stores method_shap[(n_features, shap_label)].
# We output a dataset_shap with the same structure as other datasets, but with shap_label normalized to "default".
#
# Args:
#    baseline_dataset_shap: SHAP results from all_datasets_shap_adasyn["baseline"]
#    sex_label: "baseline_male" or "baseline_female"
#    top_n_feats_candidates: Vector of feature counts (e.g., [10, 20, 30, ...])
#
# Returns:
#    A Dict with the same structure as dataset_shap for other datasets
# ============================================================================
function materialize_baseline_sex_shap(
    baseline_dataset_shap::Dict,
    sex_label::String,
    top_n_feats_candidates::Vector{Int}
)::Dict
    
    # Validate input
    @assert sex_label in ["baseline_male", "baseline_female"] "sex_label must be 'baseline_male' or 'baseline_female'"
    
    # Initialize output dictionary
    dataset_shap_out = Dict()
    
    # Loop through methods
    for (method, method_shap_in) in baseline_dataset_shap
        method_shap_out = Dict()
        
        # Loop through feature counts
        for n_features in top_n_feats_candidates
            # Check if this combination exists in the baseline SHAP
            if haskey(method_shap_in, (n_features, sex_label))
                # Map to "default" label for consistency with other datasets
                method_shap_out[(n_features, "default")] = method_shap_in[(n_features, sex_label)]
            end
        end
        
        dataset_shap_out[method] = method_shap_out
    end
    
    return dataset_shap_out
end

# ============================================================================
# Create Subject-Level Predictions Database (Do Once, Use Many Times)
#
# Create a database of subject-level predictions for all configurations.
# This can be saved to JLD2 and reused without retraining.
#
# Args:
#       all_results_adasyn: Full results dictionary
#       subjects_map: Map from dataset name to subjects vector
#       n_outer_folds: Number of outer folds (default: 5)
#       random_seed: Random seed (default: 42)
#    
# Returns:
#    DataFrame with subject-level predictions for every configuration
# ============================================================================
function create_subject_level_database(
    all_results_adasyn::Dict,
    subjects_map::Dict;
    n_outer_folds::Int = 5,
    random_seed::Int = 42
)::DataFrame
    
    println("\n" * "="^70)
    println("Creating subject-level predictions database...")
    println("="^70)
    
    results_df = DataFrame(
        Dataset = String[],
        Method = String[],
        N_Features = Int[],
        Classifier = String[],
        MCC = Float64[],
        F1 = Float64[],
        Sensitivity = Float64[],
        Specificity = Float64[],
        BACC = Float64[],
        Accuracy = Float64[],
        subject_ids = Vector{Vector{Int}}(),
        y_true = Vector{Vector{Int}}(),
        y_pred = Vector{Vector{Int}}()
    )
    
    for (dataset_name, nested_cv_results) in all_results_adasyn
        
        # Get subjects for this dataset
        if !haskey(subjects_map, dataset_name)
            println("  ⚠ Skipping $dataset_name - no subjects mapping")
            continue
        end
        
        subjects = subjects_map[dataset_name]
        
        println("  Processing: $dataset_name")
        
        # Get all methods, feature counts, classifiers from first fold
        first_fold = nested_cv_results[1]
        methods = collect(keys(first_fold))
        
        for method in methods
            # Skip jaccard_indices - these are similarity scores, not classifier results
            if method == "jaccard_indices"
                continue
            end
            
            method_results = first_fold[method]
            feature_count_keys = collect(keys(method_results))
            
            for feature_count_key in feature_count_keys
                # Extract number from feature count key
                n_features = nothing
                try
                    # Convert to string
                    key_str = string(feature_count_key)
                    
                    # Handle different formats:
                    # "10_features" -> 10
                    # "selected_features_40" -> 40
                    # "40" -> 40
                    if occursin("selected_features_", key_str)
                        # Format: "selected_features_40" -> extract "40"
                        parts = split(key_str, "_")
                        n_features = parse(Int, parts[end])  # Take last part
                    elseif occursin("_features", key_str)
                        # Format: "10_features" -> extract "10"
                        parts = split(key_str, "_")
                        n_features = parse(Int, parts[1])  # Take first part
                    else
                        # Maybe it's just a number as string or symbol
                        n_features = parse(Int, key_str)
                    end
                catch e
                    println("    ⚠ Could not parse feature count from key: $feature_count_key")
                    continue
                end
                
                if isnothing(n_features)
                    continue
                end
                
                classifier_results = method_results[feature_count_key]
                classifiers = collect(keys(classifier_results))
                
                for classifier in classifiers
                    # Skip sex-stratified classifiers - they have nested structure
                    if endswith(string(classifier), "_sex")
                        continue
                    end
                    
                    # Collect subject-level predictions across folds
                    all_subject_ids = Int[]
                    all_y_true = Int[]
                    all_y_pred = Int[]
                    
                    try
                        # Process each fold
                        for fold_idx in 1:n_outer_folds
                            if fold_idx > length(nested_cv_results)
                                break
                            end
                            
                            # Check if configuration exists in this fold
                            if !haskey(nested_cv_results[fold_idx], method)
                                continue
                            end
                            if !haskey(nested_cv_results[fold_idx][method], feature_count_key)
                                continue
                            end
                            if !haskey(nested_cv_results[fold_idx][method][feature_count_key], classifier)
                                continue
                            end
                            
                            # Get recording-level results
                            result = nested_cv_results[fold_idx][method][feature_count_key][classifier]
                            
                            # Check if it has the expected structure
                            if !hasfield(typeof(result), :y_true) || !hasfield(typeof(result), :y_pred)
                                continue
                            end
                            
                            y_true_rec = result.y_true
                            y_pred_rec = result.y_pred
                            
                            # Figure out which subjects were tested in this fold
                            n_test_recordings = length(y_true_rec)
                            
                            # For each subject, aggregate their 3 recordings
                            # Group consecutive recordings by 3
                            for i in 1:3:n_test_recordings
                                if i + 2 > n_test_recordings
                                    break  # Skip incomplete triplets
                                end
                                
                                # Get 3 recordings for this subject
                                subj_y_true = y_true_rec[i:i+2]
                                subj_y_pred = y_pred_rec[i:i+2]
                                
                                # Get true label (use most common if inconsistent)
                                true_label = mode(subj_y_true)
                                
                                # Subject-level aggregation via majority vote
                                subject_id = div(i-1, 3) + 1 + length(all_subject_ids)
                                push!(all_subject_ids, subject_id)
                                push!(all_y_true, true_label)
                                push!(all_y_pred, majority_vote(subj_y_pred))
                            end
                        end
                        
                        # Only add if we got valid predictions
                        if length(all_y_true) == 0
                            continue
                        end
                        
                        # Compute subject-level metrics
                        mcc, f1, sen, spec, bacc, acc = calc_perf_eval_measures(
                            all_y_true, all_y_pred
                        )
                        
                        # Add to database
                        push!(results_df, (
                            Dataset = dataset_name,
                            Method = method,
                            N_Features = n_features,
                            Classifier = classifier,
                            MCC = mcc,
                            F1 = f1,
                            Sensitivity = sen,
                            Specificity = spec,
                            BACC = bacc,
                            Accuracy = acc,
                            subject_ids = all_subject_ids,
                            y_true = all_y_true,
                            y_pred = all_y_pred
                        ))
                        
                    catch e
                        # Silently skip - already handled above
                        continue
                    end
                end
            end
        end
        
        println("    ✓ Complete")
    end
    
    println("\n✓ Database created: $(nrow(results_df)) configurations")
    println("="^70)
    
    return results_df
end

# ============================================================================
# Export Predictions with Fold Information and Class Labels
#
# Export subject-level predictions in detailed format with fold information.
# Converts 0/1 predictions to PD/HC labels.
# ============================================================================
function export_predictions_detailed(
    subject_level_db::DataFrame;
    filename::String = "subject_predictions_detailed_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv",
    n_outer_folds::Int = 5
)
    
    println("\n" * "="^70)
    println("Exporting detailed predictions with fold information...")
    println("="^70)
    
    # Create long-format dataframe
    predictions_df = DataFrame(
        Subject_ID = Int[],
        Outer_Fold = Int[],
        Model_Configuration = String[],
        y_pred = String[],
        y_true = String[]
    )
    
    # Class label mapping (assuming 0=HC, 1=PD)
    label_map = Dict(0 => "HC", 1 => "PD")
    
    for row in eachrow(subject_level_db)
        # Create readable model configuration string
        model_config = "$(row.Dataset) ($(row.Method) - $(row.N_Features) - $(row.Classifier))"
        
        # Total number of subjects
        n_subjects = length(row.y_true)
        
        # Calculate subjects per fold (approximately)
        subjects_per_fold = ceil(Int, n_subjects / n_outer_folds)
        
        # Assign each subject to a fold and local subject ID
        for (global_idx, (yt, yp)) in enumerate(zip(row.y_true, row.y_pred))
            # Determine which fold this subject is in
            outer_fold = min(ceil(Int, global_idx / subjects_per_fold), n_outer_folds)
            
            # Subject ID within the fold (1-indexed)
            subject_id_in_fold = mod1(global_idx, subjects_per_fold)
            
            # Convert labels
            y_true_label = label_map[yt]
            y_pred_label = label_map[yp]
            
            push!(predictions_df, (
                Subject_ID = subject_id_in_fold,
                Outer_Fold = outer_fold,
                Model_Configuration = model_config,
                y_pred = y_pred_label,
                y_true = y_true_label
            ))
        end
    end
    
    # Sort for readability
    sort!(predictions_df, [:Model_Configuration, :Outer_Fold, :Subject_ID])
    
    # Export
    CSV.write(filename, predictions_df)
    
    println("✓ Exported $(nrow(predictions_df)) predictions")
    println("  File: $filename")
    println("  Configurations: $(length(unique(predictions_df.Model_Configuration)))")
    println("  Total subjects: $(length(unique(predictions_df.Subject_ID)))")
    println("="^70)

    return predictions_df
end

# ============================================================================
# INNER-FOLD METRIC EXTRACTION (Post-hoc second pass)
#
# Reconstructs the exact inner folds used during training and evaluates every
# classifier on both the inner validation set and the pre-ADASYN inner training
# set.  Outer test metrics are appended (Inner_Fold = 0, Split = "test") so
# that train / val / test curves can be built from a single DataFrame.
#
# Selected features and best-model hyperparameters are read from all_results;
# no feature selection or grid search is repeated.
#
# Arguments
#   all_results           — nested dict from the main CV run
#   dataset_map           — Dict{String => (X, y, subjects)} for core datasets
#   top_n_feats_candidates — vector of feature counts (e.g. [10, 20, ...])
#   n_outer_folds         — number of outer folds
#   n_inner_folds         — number of inner folds
#   random_seed           — base random seed (same value used in training)
#
# Returns flat DataFrame with columns:
#   Dataset, Method, N_Features, Classifier, Outer_Fold, Inner_Fold,
#   Split, MCC, F1, Sensitivity, Specificity, BACC, Accuracy
# ============================================================================
function extract_inner_fold_metrics(
    all_results::Dict,
    dataset_map::Dict,
    top_n_feats_candidates::Vector{Int},
    n_outer_folds::Int,
    n_inner_folds::Int,
    random_seed::Int;
    sex_map::Dict = Dict()
)::DataFrame

    # ------------------------------------------------------------------
    # Helper: subject-level majority-vote metrics
    # ------------------------------------------------------------------
    function _subj_metrics(p_pos::Vector{Float64},
                           eval_subjects::Vector{Int},
                           y_eval::Vector{Int})
        subj_preds = Dict{Int, Vector{Int}}()
        subj_true  = Dict{Int, Int}()
        yhat = [p >= 0.5 ? 1 : 0 for p in p_pos]
        @inbounds for (i, subj_id) in enumerate(eval_subjects)
            push!(get!(()->Int[], subj_preds, subj_id), yhat[i])
            subj_true[subj_id] = y_eval[i]
        end
        unique_subjects    = sort(collect(keys(subj_true)))
        subject_level_true = [subj_true[s]                        for s in unique_subjects]
        subject_level_pred = [majority_vote(subj_preds[s])        for s in unique_subjects]
        return calc_perf_eval_measures(subject_level_true, subject_level_pred)
    end

    # ------------------------------------------------------------------
    # Helper: Gaussian Naive Bayes predict (returns row-level p_pos)
    # ------------------------------------------------------------------
    function _nb_predict(X_pred::AbstractMatrix,
                         means::Dict, stds::Dict,
                         class_priors::Dict, classes::Vector)
        n_pred    = size(X_pred, 1)
        n_feats   = size(X_pred, 2)
        n_classes = length(classes)
        probs_matrix = zeros(Float64, n_classes, n_pred)
        @inbounds for i in 1:n_pred
            x = X_pred[i, :]
            for (j, c) in enumerate(classes)
                lp = log(class_priors[c])
                for f in 1:n_feats
                    μ = means[c][f]; σ = stds[c][f]
                    lp += -0.5 * log(2π * σ^2) - ((x[f] - μ)^2) / (2 * σ^2)
                end
                probs_matrix[j, i] = lp
            end
        end
        @inbounds for i in 1:n_pred
            max_lp = maximum(probs_matrix[:, i])
            probs_matrix[:, i] = exp.(probs_matrix[:, i] .- max_lp)
            probs_matrix[:, i] ./= sum(probs_matrix[:, i])
        end
        pos_idx = findfirst(==(1), classes)
        isnothing(pos_idx) && return fill(0.0, n_pred)
        return vec(probs_matrix[pos_idx, :])
    end

    # ------------------------------------------------------------------
    # Classifier list (must match keys stored by process_outer_fold)
    # ------------------------------------------------------------------
    classifiers = [
        "RandomForest", "NeuralNetwork", "NaiveBayes",
        "LogisticRegression", "AdaBoost", "SVM_RBF",
        "RandomForest_TUNED", "AdaBoost_TUNED",
        "NeuralNetwork_TUNED", "SVM_RBF_TUNED",
    ]

    # ------------------------------------------------------------------
    # Output DataFrame
    # ------------------------------------------------------------------
    results_df = DataFrame(
        Dataset     = String[],
        Method      = String[],
        N_Features  = Int[],
        Classifier  = String[],
        Outer_Fold  = Int[],
        Inner_Fold  = Int[],
        Split       = String[],
        MCC         = Float64[],
        F1          = Float64[],
        Sensitivity = Float64[],
        Specificity = Float64[],
        BACC        = Float64[],
        Accuracy    = Float64[],
    )

    function _push_row!(df, dataset, method, n_feat, clf, o_fold, i_fold, split,
                        mcc, f1, sen, spec, bacc, acc)
        push!(df, (dataset, method, n_feat, clf, o_fold, i_fold, split,
                   mcc, f1, sen, spec, bacc, acc))
    end

    function _push_sex_val_rows!(df, p_val, subj_ival, y_ival, sex_vec,
                                 method, n_features, clf, o_fold, i_fold)
        for (sex_val, sex_ds) in ((1.0, "baseline_male"), (0.0, "baseline_female"))
            mask = sex_vec .== sex_val
            sum(mask) < 2 && continue
            length(unique(y_ival[mask])) < 2 && continue
            mcc, f1, sen, spec, bacc, acc = _subj_metrics(p_val[mask], subj_ival[mask], y_ival[mask])
            _push_row!(df, sex_ds, method, n_features, clf,
                       o_fold, i_fold, "val", mcc, f1, sen, spec, bacc, acc)
        end
    end

    # ==================================================================
    # MAIN LOOP
    # ==================================================================
    for (dataset_name, (X, y, subjects)) in dataset_map

        !haskey(all_results, dataset_name) && continue
        is_baseline = (dataset_name == "baseline")

        println("\n" * "="^60)
        println("extract_inner_fold_metrics: $dataset_name")
        println("="^60)

        outer_folds = stratified_subject_level_cv(
            subjects, y, n_outer_folds; random_seed=random_seed)

        for outer_fold_idx in 1:n_outer_folds
            fold_seed   = random_seed + outer_fold_idx
            outer_fold  = outer_folds[outer_fold_idx]

            X_outer_train      = X[outer_fold.train, :]
            y_outer_train      = y[outer_fold.train]
            subj_outer_train   = subjects[outer_fold.train]
            sex_outer_train    = is_baseline && haskey(sex_map, "baseline") ?
                                 sex_map["baseline"][outer_fold.train] : Float64[]

            inner_folds = stratified_subject_level_cv(
                subj_outer_train, y_outer_train, n_inner_folds;
                random_seed=fold_seed)

            for method in ["mRMR", "Boruta"]
                for n_features in top_n_feats_candidates
                    feature_count    = "$(n_features)_features"
                    saved_features   = all_results[dataset_name][outer_fold_idx][method]["selected_features_$(n_features)"]

                    # ---- append outer TEST metrics (no recomputation) ----
                    for clf in classifiers
                        !haskey(all_results[dataset_name][outer_fold_idx][method][feature_count], clf) && continue
                        r = all_results[dataset_name][outer_fold_idx][method][feature_count][clf]
                        _push_row!(results_df, dataset_name, method, n_features, clf,
                                   outer_fold_idx, 0, "test",
                                   r.mcc, r.f1, r.sen, r.spec, r.bacc, r.acc)
                        if is_baseline
                            sex_key = "$(clf)_sex"
                            fdict = all_results[dataset_name][outer_fold_idx][method][feature_count]
                            if haskey(fdict, sex_key)
                                sr = fdict[sex_key]
                                for (sex_res, sex_ds) in ((sr.male, "baseline_male"), (sr.female, "baseline_female"))
                                    isnan(sex_res.mcc) && continue
                                    _push_row!(results_df, sex_ds, method, n_features, clf,
                                               outer_fold_idx, 0, "test",
                                               sex_res.mcc, sex_res.f1, sex_res.sen,
                                               sex_res.spec, sex_res.bacc, sex_res.acc)
                                end
                            end
                        end
                    end

                    # ---- inner fold loop ----
                    for inner_fold_idx in 1:n_inner_folds
                        inner_seed  = fold_seed + inner_fold_idx
                        inner_fold  = inner_folds[inner_fold_idx]

                        # Feature-selected raw slices
                        X_itr_raw  = X_outer_train[inner_fold.train, saved_features]
                        y_itr      = y_outer_train[inner_fold.train]
                        subj_itr   = subj_outer_train[inner_fold.train]

                        X_ival_raw = X_outer_train[inner_fold.test, saved_features]
                        y_ival     = y_outer_train[inner_fold.test]
                        subj_ival  = subj_outer_train[inner_fold.test]
                        sex_ival   = is_baseline && !isempty(sex_outer_train) ?
                                     sex_outer_train[inner_fold.test] : Float64[]

                        # Standardize (fit on inner train)
                        X_itr_S, X_ival_S = standardize_pair(X_itr_raw, X_ival_raw)

                        # ADASYN on standardized inner train
                        X_bal, y_bal = resample_training_data(
                            X_itr_S, y_itr; rng_seed=inner_seed)

                        n_feats_sel = size(X_bal, 2)

                        # ---- evaluate each classifier ----
                        for clf in classifiers

                            # Try/catch so a single failure doesn't abort the loop
                            try

                            p_val = Float64[]
                            # ---- RandomForest (untuned) ----
                            if clf == "RandomForest"
                                rf = MLJDecisionTreeInterface.RandomForestClassifier(
                                        n_trees=100, rng=inner_seed)
                                mach = machine(rf, MLJ.table(X_bal), categorical(y_bal))
                                MLJ.fit!(mach, verbosity=0)
                                for (Xev, yev, subjev, split) in [
                                        (X_ival_S, y_ival, subj_ival, "val"),
                                        (X_itr_S,  y_itr,  subj_itr,  "train")]
                                    yp  = MLJ.predict(mach, MLJ.table(Xev))
                                    p   = [Float64(pdf(s, 1)) for s in yp]
                                    mcc, f1, sen, spec, bacc, acc = _subj_metrics(p, subjev, yev)
                                    _push_row!(results_df, dataset_name, method, n_features, clf,
                                               outer_fold_idx, inner_fold_idx, split,
                                               mcc, f1, sen, spec, bacc, acc)
                                    split == "val" && (p_val = p)
                                end

                            # ---- NeuralNetwork (untuned) ----
                            elseif clf == "NeuralNetwork"
                                nn = nn_builder(n_feats_sel)
                                mach = machine(nn, MLJ.table(X_bal), categorical(y_bal))
                                MLJ.fit!(mach, verbosity=0)
                                for (Xev, yev, subjev, split) in [
                                        (X_ival_S, y_ival, subj_ival, "val"),
                                        (X_itr_S,  y_itr,  subj_itr,  "train")]
                                    yp  = MLJ.predict(mach, MLJ.table(Xev))
                                    p   = [Float64(pdf(s, 1)) for s in yp]
                                    mcc, f1, sen, spec, bacc, acc = _subj_metrics(p, subjev, yev)
                                    _push_row!(results_df, dataset_name, method, n_features, clf,
                                               outer_fold_idx, inner_fold_idx, split,
                                               mcc, f1, sen, spec, bacc, acc)
                                    split == "val" && (p_val = p)
                                end

                            # ---- NaiveBayes (manual Gaussian) ----
                            elseif clf == "NaiveBayes"
                                classes = unique(y_bal)
                                nb_priors = Dict(c => sum(y_bal .== c) / length(y_bal) for c in classes)
                                nb_means  = Dict{Int, Vector{Float64}}()
                                nb_stds   = Dict{Int, Vector{Float64}}()
                                for c in classes
                                    mask = y_bal .== c
                                    nb_means[c] = vec(mean(X_bal[mask, :], dims=1))
                                    nb_stds[c]  = vec(std(X_bal[mask, :],  dims=1))
                                    nb_stds[c] .= max.(nb_stds[c], 1e-10)
                                end
                                for (Xev, yev, subjev, split) in [
                                        (X_ival_S, y_ival, subj_ival, "val"),
                                        (X_itr_S,  y_itr,  subj_itr,  "train")]
                                    p   = _nb_predict(Xev, nb_means, nb_stds, nb_priors, classes)
                                    mcc, f1, sen, spec, bacc, acc = _subj_metrics(p, subjev, yev)
                                    _push_row!(results_df, dataset_name, method, n_features, clf,
                                               outer_fold_idx, inner_fold_idx, split,
                                               mcc, f1, sen, spec, bacc, acc)
                                    split == "val" && (p_val = p)
                                end

                            # ---- LogisticRegression ----
                            elseif clf == "LogisticRegression"
                                train_df  = DataFrame(X_bal, :auto)
                                train_df.y = y_bal
                                formula   = Term(:y) ~ sum(Term.(Symbol.(names(train_df, Not(:y)))))
                                lr_model  = try
                                    glm(formula, train_df, Binomial(), LogitLink();
                                        maxiter=100, atol=1e-6, rtol=1e-6)
                                catch
                                    nothing
                                end
                                for (Xev, yev, subjev, split) in [
                                        (X_ival_S, y_ival, subj_ival, "val"),
                                        (X_itr_S,  y_itr,  subj_itr,  "train")]
                                    test_df = DataFrame(Xev, :auto)
                                    p = if !isnothing(lr_model)
                                        Float64.(coalesce.(GLM.predict(lr_model, test_df), 0.5))
                                    else
                                        fill(mean(y_bal), size(Xev, 1))
                                    end
                                    mcc, f1, sen, spec, bacc, acc = _subj_metrics(p, subjev, yev)
                                    _push_row!(results_df, dataset_name, method, n_features, clf,
                                               outer_fold_idx, inner_fold_idx, split,
                                               mcc, f1, sen, spec, bacc, acc)
                                    split == "val" && (p_val = p)
                                end

                            # ---- AdaBoost (untuned) ----
                            elseif clf == "AdaBoost"
                                ada = AdaBoostModel(n_iter=50)
                                mach = machine(ada, MLJ.table(X_bal), categorical(y_bal))
                                MLJ.fit!(mach, verbosity=0)
                                for (Xev, yev, subjev, split) in [
                                        (X_ival_S, y_ival, subj_ival, "val"),
                                        (X_itr_S,  y_itr,  subj_itr,  "train")]
                                    yp  = MLJ.predict(mach, MLJ.table(Xev))
                                    p   = [Float64(pdf(s, 1)) for s in yp]
                                    mcc, f1, sen, spec, bacc, acc = _subj_metrics(p, subjev, yev)
                                    _push_row!(results_df, dataset_name, method, n_features, clf,
                                               outer_fold_idx, inner_fold_idx, split,
                                               mcc, f1, sen, spec, bacc, acc)
                                    split == "val" && (p_val = p)
                                end

                            # ---- SVM_RBF (untuned) — string categorical targets ----
                            elseif clf == "SVM_RBF"
                                svm_m = SVM(kernel=LIBSVM.Kernel.RadialBasis, gamma=0.02, cost=20.0)
                                y_cat = categorical(string.(y_bal), levels=["0","1"])
                                mach  = MLJ.machine(svm_m, MLJ.table(X_bal), y_cat)
                                MLJ.fit!(mach, verbosity=0)
                                for (Xev, yev, subjev, split) in [
                                        (X_ival_S, y_ival, subj_ival, "val"),
                                        (X_itr_S,  y_itr,  subj_itr,  "train")]
                                    yp  = MLJ.predict(mach, MLJ.table(Xev))
                                    p   = [Float64(pdf(s, "1")) for s in yp]
                                    mcc, f1, sen, spec, bacc, acc = _subj_metrics(p, subjev, yev)
                                    _push_row!(results_df, dataset_name, method, n_features, clf,
                                               outer_fold_idx, inner_fold_idx, split,
                                               mcc, f1, sen, spec, bacc, acc)
                                    split == "val" && (p_val = p)
                                end

                            # ---- RandomForest_TUNED — reuse saved best hyperparams ----
                            elseif clf == "RandomForest_TUNED"
                                bm  = all_results[dataset_name][outer_fold_idx][method][feature_count]["RandomForest_TUNED"].best_model
                                rf_t = MLJDecisionTreeInterface.RandomForestClassifier(
                                    n_trees=bm.n_trees, max_depth=bm.max_depth, rng=inner_seed)
                                mach = machine(rf_t, MLJ.table(X_bal), categorical(y_bal))
                                MLJ.fit!(mach, verbosity=0)
                                for (Xev, yev, subjev, split) in [
                                        (X_ival_S, y_ival, subj_ival, "val"),
                                        (X_itr_S,  y_itr,  subj_itr,  "train")]
                                    yp  = MLJ.predict(mach, MLJ.table(Xev))
                                    p   = [Float64(pdf(s, 1)) for s in yp]
                                    mcc, f1, sen, spec, bacc, acc = _subj_metrics(p, subjev, yev)
                                    _push_row!(results_df, dataset_name, method, n_features, clf,
                                               outer_fold_idx, inner_fold_idx, split,
                                               mcc, f1, sen, spec, bacc, acc)
                                    split == "val" && (p_val = p)
                                end

                            # ---- AdaBoost_TUNED ----
                            elseif clf == "AdaBoost_TUNED"
                                bm   = all_results[dataset_name][outer_fold_idx][method][feature_count]["AdaBoost_TUNED"].best_model
                                ada_t = AdaBoostModel(n_iter=bm.n_iter)
                                mach  = machine(ada_t, MLJ.table(X_bal), categorical(y_bal))
                                MLJ.fit!(mach, verbosity=0)
                                for (Xev, yev, subjev, split) in [
                                        (X_ival_S, y_ival, subj_ival, "val"),
                                        (X_itr_S,  y_itr,  subj_itr,  "train")]
                                    yp  = MLJ.predict(mach, MLJ.table(Xev))
                                    p   = [Float64(pdf(s, 1)) for s in yp]
                                    mcc, f1, sen, spec, bacc, acc = _subj_metrics(p, subjev, yev)
                                    _push_row!(results_df, dataset_name, method, n_features, clf,
                                               outer_fold_idx, inner_fold_idx, split,
                                               mcc, f1, sen, spec, bacc, acc)
                                    split == "val" && (p_val = p)
                                end

                            # ---- NeuralNetwork_TUNED — integer targets ----
                            elseif clf == "NeuralNetwork_TUNED"
                                bm   = all_results[dataset_name][outer_fold_idx][method][feature_count]["NeuralNetwork_TUNED"].best_model
                                nn_t = nn_builder(n_feats_sel)
                                nn_t.lambda = bm.lambda
                                nn_t.alpha  = bm.alpha
                                mach = machine(nn_t, MLJ.table(X_bal), categorical(y_bal))
                                MLJ.fit!(mach, verbosity=0)
                                for (Xev, yev, subjev, split) in [
                                        (X_ival_S, y_ival, subj_ival, "val"),
                                        (X_itr_S,  y_itr,  subj_itr,  "train")]
                                    yp  = MLJ.predict(mach, MLJ.table(Xev))
                                    p   = [Float64(pdf(s, 1)) for s in yp]
                                    mcc, f1, sen, spec, bacc, acc = _subj_metrics(p, subjev, yev)
                                    _push_row!(results_df, dataset_name, method, n_features, clf,
                                               outer_fold_idx, inner_fold_idx, split,
                                               mcc, f1, sen, spec, bacc, acc)
                                    split == "val" && (p_val = p)
                                end

                            # ---- SVM_RBF_TUNED — string targets, Pipeline best_model ----
                            elseif clf == "SVM_RBF_TUNED"
                                bm    = all_results[dataset_name][outer_fold_idx][method][feature_count]["SVM_RBF_TUNED"].best_model
                                pipe_t = Pipeline(
                                    svm = SVM(kernel=LIBSVM.Kernel.RadialBasis,
                                              cost=bm.svm.cost, gamma=bm.svm.gamma);
                                    prediction_type=:probabilistic)
                                y_cat  = categorical(string.(y_bal), levels=["0","1"])
                                mach   = machine(pipe_t, MLJ.table(X_bal), y_cat)
                                MLJ.fit!(mach, verbosity=0)
                                for (Xev, yev, subjev, split) in [
                                        (X_ival_S, y_ival, subj_ival, "val"),
                                        (X_itr_S,  y_itr,  subj_itr,  "train")]
                                    yp  = MLJ.predict(mach, MLJ.table(Xev))
                                    p   = [Float64(pdf(s, "1")) for s in yp]
                                    mcc, f1, sen, spec, bacc, acc = _subj_metrics(p, subjev, yev)
                                    _push_row!(results_df, dataset_name, method, n_features, clf,
                                               outer_fold_idx, inner_fold_idx, split,
                                               mcc, f1, sen, spec, bacc, acc)
                                    split == "val" && (p_val = p)
                                end

                            end  # classifier dispatch

                            if is_baseline && !isempty(p_val)
                                _push_sex_val_rows!(results_df, p_val, subj_ival, y_ival, sex_ival,
                                                    method, n_features, clf, outer_fold_idx, inner_fold_idx)
                            end

                            catch e
                                @warn "extract_inner_fold_metrics: skipped $clf " *
                                      "($dataset_name, $method, $(n_features)f, " *
                                      "outer=$outer_fold_idx, inner=$inner_fold_idx): $e"
                            end

                        end  # for clf
                    end  # for inner_fold_idx
                end  # for n_features
            end  # for method
        end  # for outer_fold_idx

        println("  ✓ $dataset_name complete")
    end  # for dataset

    println("\n✓ extract_inner_fold_metrics: $(nrow(results_df)) rows total")
    return results_df
end


# ============================================================================
# SUPPLEMENTAL TABLE — inner fold validation metrics summary
#
# Aggregates the val-split rows from extract_inner_fold_metrics into a
# publication-ready table with mean ± 95% CI for each metric.
# Saved to CSV and rendered via PrettyTables.
#
# Arguments
#   inner_fold_df — output of extract_inner_fold_metrics
#   strategy      — label appended to the output filename
# ============================================================================
function generate_supplemental_table(
    inner_fold_df::DataFrame,
    strategy::String = "ADASYN";
    output_dir::String = "."
)
    val_df = filter(r -> r.Split == "val", inner_fold_df)

    # Columns: Dataset, Method, N_Features, Classifier, then metrics
    metric_cols = [:MCC, :F1, :Sensitivity, :Specificity, :BACC, :Accuracy]

    rows = DataFrame(
        Dataset    = String[],
        Method     = String[],
        N_Features = Int[],
        Classifier = String[],
        MCC        = String[],
        F1         = String[],
        Accuracy   = String[],
        BACC       = String[],
        Sensitivity = String[],
        Specificity = String[],
    )

    for gdf in groupby(val_df, [:Dataset, :Method, :N_Features, :Classifier])
        key = first(gdf)
        n   = nrow(gdf)
        n < 2 && continue   # skip degenerate groups

        t_crit = quantile(TDist(n - 1), 0.975)

        fmt_ci(col) = begin
            vals = Float64.(gdf[!, col])
            m  = mean(vals)
            ci = t_crit * std(vals) / sqrt(n)
            @sprintf("%.3f ± %.3f", m, ci)
        end

        push!(rows, (
            Dataset     = key.Dataset,
            Method      = key.Method,
            N_Features  = key.N_Features,
            Classifier  = key.Classifier,
            MCC         = fmt_ci(:MCC),
            F1          = fmt_ci(:F1),
            Accuracy    = fmt_ci(:Accuracy),
            BACC        = fmt_ci(:BACC),
            Sensitivity = fmt_ci(:Sensitivity),
            Specificity = fmt_ci(:Specificity),
        ))
    end

    sort!(rows, [:Dataset, :Method, :N_Features, :Classifier])

    # Save CSV
    date     = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename = joinpath(output_dir, "supplemental_inner_val_metrics_$(strategy)_$(date).csv")
    CSV.write(filename, rows)

    # Render
    println("\n" * "="^80)
    println("Supplemental Table — Inner Fold Validation Metrics (mean ± 95% CI)")
    println("Strategy: $strategy")
    println("="^80 * "\n")

    hl = Highlighter(
        (data, i, j) -> false,   # no conditional highlight — uniform table
        Crayon(foreground=:default)
    )

    pretty_table(rows,
        header      = ["Dataset", "Method", "N Feat", "Classifier",
                        "MCC", "F1", "Accuracy", "BACC", "Sensitivity", "Specificity"],
        alignment   = [:l, :l, :r, :l, :r, :r, :r, :r, :r, :r],
        crop        = :none,
        autowrap    = true,
        show_subheader = false)

    println("\n✓ Saved: $filename  ($(nrow(rows)) configurations)")
    println("="^80)

    return rows
end