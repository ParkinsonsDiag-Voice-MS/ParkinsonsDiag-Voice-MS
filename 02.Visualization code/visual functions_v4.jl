# --- set default font style ----
default(fontfamily="Arial")

# --- helper: hex -> RGBA with alpha ---
hex_rgba(hex::AbstractString, a::Real) = begin
    h = replace(hex, "#" => "")
    r = parse(Int, h[1:2], base=16) / 255
    g = parse(Int, h[3:4], base=16) / 255
    b = parse(Int, h[5:6], base=16) / 255
    RGBA(r, g, b, a)
end

# ============================================================================
# Create comparison plot directly from nested_cv_results structure.
#    Works with sex-stratified results embedded in the nested structure.    
#    Parameters:
#    - all_results: Dict with keys like "baseline", "male", "female", "sex aware"
#    - top_n_feats_candidates: Vector of feature counts tested
# ============================================================================
function quick_mcc_comparison_from_nested_results(
    all_results::Dict,
    top_n_feats_candidates::Vector{Int}
)

    # -------------------------------------------------------------------------
    # What to compare
    # -------------------------------------------------------------------------
    datasets = ["baseline", "baseline_male", "baseline_female", "sex aware", "male", "female"]
    methods  = ["mRMR", "Boruta", "Pearson"]
    classifiers = [
        "RandomForest", "RandomForest_TUNED",
        "NeuralNetwork", "NeuralNetwork_TUNED",
        "NaiveBayes", "LogisticRegression",
        "AdaBoost", "AdaBoost_TUNED",
        "SVM_RBF", "SVM_RBF_TUNED"
    ]

    # -------------------------------------------------------------------------
    # Store best results
    # -------------------------------------------------------------------------
    best_results = DataFrame(
        Dataset    = String[],
        Classifier = String[],
        Method     = String[],
        Features   = Int[],
        MCC        = Float64[]
    )

    # -------------------------------------------------------------------------
    # Find best config per dataset
    # -------------------------------------------------------------------------
    for dataset in datasets
        best_mcc    = -Inf
        best_config = nothing

        if dataset == "baseline_male" || dataset == "baseline_female"
            if !haskey(all_results, "baseline")
                println("Warning: No baseline results found")
                continue
            end
            nested_cv_results = all_results["baseline"]
            sex_filter = dataset == "baseline_male" ? "male" : "female"
        else
            if !haskey(all_results, dataset)
                println("Warning: No data found for dataset '$dataset'")
                continue
            end
            nested_cv_results = all_results[dataset]
            sex_filter = nothing
        end

        for method in methods
            for n_features in top_n_feats_candidates
                feature_count = "$(n_features)_features"

                for classifier in classifiers
                    mcc_values = Float64[]

                    for outer_fold_idx in 1:length(nested_cv_results)
                        fold_results = nested_cv_results[outer_fold_idx]

                        if !haskey(fold_results, method) || !haskey(fold_results[method], feature_count)
                            continue
                        end

                        if sex_filter !== nothing
                            sex_key = "$(classifier)_sex"
                            if haskey(fold_results[method][feature_count], sex_key)
                                sex_results = fold_results[method][feature_count][sex_key]

                                mcc = sex_filter == "male" ? sex_results.male.mcc : sex_results.female.mcc
                                if !isnan(mcc)
                                    push!(mcc_values, mcc)
                                end
                            end
                        else
                            if haskey(fold_results[method][feature_count], classifier)
                                mcc = fold_results[method][feature_count][classifier].mcc
                                if !isnan(mcc)
                                    push!(mcc_values, mcc)
                                end
                            end
                        end
                    end

                    if !isempty(mcc_values)
                        mean_mcc = mean(mcc_values)
                        if mean_mcc > best_mcc
                            best_mcc = mean_mcc
                            best_config = (
                                Dataset    = dataset,
                                Classifier = classifier,
                                Method     = method,
                                Features   = n_features,
                                MCC        = mean_mcc
                            )
                        end
                    end
                end
            end
        end

        if best_config !== nothing
            push!(best_results, best_config)
        end
    end

    # -------------------------------------------------------------------------
    # X-axis labels
    # -------------------------------------------------------------------------
    label_map = Dict(
        "baseline"        => "Full Dataset\n(All)",
        "baseline_male"   => "Full Dataset\n(Male Evaluation)",
        "baseline_female" => "Full Dataset\n(Female Evaluation)",
        "sex aware"       => "Full Dataset\n(Sex-Aware)",
        "male"            => "Male-Only\nModel",
        "female"          => "Female-Only\nModel"
    )
    x_labels = [label_map[d] for d in best_results.Dataset]

    # -------------------------------------------------------------------------
    # Annotation label maps
    # -------------------------------------------------------------------------
    clf_label = Dict(
        "RandomForest"         => "Random Forest",
        "RandomForest_TUNED"   => "Random Forest (tuned)",
        "NeuralNetwork"        => "Neural Network",
        "NeuralNetwork_TUNED"  => "Neural Network (tuned)",
        "NaiveBayes"           => "Naive Bayes",
        "LogisticRegression"   => "Logistic\nRegression",
        "AdaBoost"             => "AdaBoost",
        "AdaBoost_TUNED"       => "AdaBoost (tuned)",
        "SVM_RBF"              => "SVM-RBF",
        "SVM_RBF_TUNED"        => "SVM-RBF (tuned)"
    )
    method_label = Dict("mRMR" => "mRMR", "Boruta" => "Boruta", "Pearson" => "Pearson")

    # -------------------------------------------------------------------------
    # Colors (fills + outlines)
    # -------------------------------------------------------------------------
    DATASET_COLORS = Dict(
        "baseline"        => "#785ef0",
        "baseline_male"   => hex_rgba("#2C2796", 0.55),
        "baseline_female" => hex_rgba("#973DA4", 0.55),
        "sex aware"       => "#ffb000",
        "male"            => "#648fff",
        "female"          => "#dc267f"
    )
    bar_fill = [DATASET_COLORS[d] for d in best_results.Dataset]

    black_solid = RGBA(0, 0, 0, 1.0)
	# black_solid = hex_rgba("#ffb000", 1.0)
    # blue_line   = hex_rgba("#007aa8", 1.0)  # bar 2 outline
    # pink_line   = hex_rgba("#E75481", 1.0)  # bar 3 outline

	    blue_line   = hex_rgba("#21211F", 1.0)  # bar 2 outline
    pink_line   = hex_rgba("#21211F", 1.0)  # bar 3 outline

    edge_cols = fill(black_solid, nrow(best_results))
    if nrow(best_results) ≥ 2; edge_cols[2] = blue_line; end
    if nrow(best_results) ≥ 3; edge_cols[3] = pink_line; end

    # -------------------------------------------------------------------------
    # Base bar plot (uniform outline width)
    # -------------------------------------------------------------------------
    p = bar(
        1:nrow(best_results),
        best_results.MCC;
        label="",
        xlabel = "\nModel Configuration",
        ylabel = "\n\nBest MCC Achieved",
        # title  = "Best MCC by Model Configuration",
        seriescolor = bar_fill,
        linecolor   = edge_cols,
        linewidth   = 3.0,
        xticks      = (1:nrow(best_results), x_labels),
        xrotation   = 0,
        ylim        = (0, 0.8),
        size        = (950, 600),
        bottom_margin = 10mm,

		    tickfontsize  = 12,
    guidefontsize = 12,
    # titlefontsize = 14,
    # legendfontsize = 12
    )

    # -------------------------------------------------------------------------
    # Overlay thicker outlines ONLY for bars 2 & 3
    # (Plots.jl linewidth is per-series, so we add a second bar series with
    # transparent fill at x=2,3.)
    # -------------------------------------------------------------------------
    x_thick = Int[]
    y_thick = Float64[]
    lc_thick = Any[]

    if nrow(best_results) ≥ 2
        push!(x_thick, 2); push!(y_thick, best_results.MCC[2]); push!(lc_thick, edge_cols[2])
    end
    if nrow(best_results) ≥ 3
        push!(x_thick, 3); push!(y_thick, best_results.MCC[3]); push!(lc_thick, edge_cols[3])
    end

    if !isempty(x_thick)
        bar!(
            p,
            x_thick,
            y_thick;
            label="",
            seriescolor = RGBA(0,0,0,0),  # transparent fill (outline only)
            linecolor   = lc_thick,
            linewidth   = 3.0             # <- thicker border for bars 2 & 3
        )
    end

    # -------------------------------------------------------------------------
    # Annotate bars
    # -------------------------------------------------------------------------
    for (i, row) in enumerate(eachrow(best_results))
        ann_text = @sprintf("%s\n%s\nn=%d\nMCC=%.3f",
            get(clf_label, row.Classifier, row.Classifier),
            get(method_label, row.Method, row.Method),
            row.Features,
            row.MCC
        )
        # y_pos = min(row.MCC + 0.065, 0.78)
        # annotate!(p, i, y_pos, text(ann_text, 8, :black, :center, :bold))
		y_pos = min(row.MCC + 0.1, 0.78)
		annotate!(p, i, y_pos, Plots.text(ann_text, 12, :center, :top, :black, "Helvetica Bold"))
# or try: "DejaVu Sans Bold"

    end

    return p, best_results
end

# =================================================================================
#    extract_jaccard_indices(all_results, dataset_order, top_n_feats_candidates; verbose=true)
#
# Extract pre-computed Jaccard indices from nested cross-validation results.
#
# Arguments
#   - `all_results`: Dictionary containing results for all datasets
#   - `dataset_order`: Vector of dataset names to process
#   - `top_n_feats_candidates`: Vector of feature count values to extract
#   - `verbose`: Whether to print extraction progress (default: true)
# Returns
#   - Dictionary with structure: `jaccard_all_datasets[dataset][method_pair][n_features] = [values...]`
#       where method_pair is one of: "mRMR_Boruta", "mRMR_Pearson", "Boruta_Pearson"
# =================================================================================
function extract_jaccard_indices(all_results, dataset_order, top_n_feats_candidates; 
    verbose=true)
    
    jaccard_all_datasets = Dict()
    method_pairs = ["mRMR_Boruta", "mRMR_Pearson", "Boruta_Pearson"]
    
    for dataset in dataset_order
        verbose && println("Extracting Jaccard indices for: $dataset")
        jaccard_all_datasets[dataset] = Dict()
        
        if !haskey(all_results, dataset)
            verbose && println("  ⚠ Dataset not found")
            continue
        end
        
        dataset_results = all_results[dataset]
        n_folds = length(dataset_results)
        
        # Initialize storage for method pairs
        for pair in method_pairs
            jaccard_all_datasets[dataset][pair] = Dict()
            
            for n_feat in top_n_feats_candidates
                jaccard_all_datasets[dataset][pair][n_feat] = Float64[]
            end
        end
        
        # Extract from each fold
        for fold_idx in 1:n_folds
            fold_data = dataset_results[fold_idx]
            
            if haskey(fold_data, "jaccard_indices")
                jacc_data = fold_data["jaccard_indices"]
                
                # Iterate through feature counts
                for n_feat in top_n_feats_candidates
                    if haskey(jacc_data, n_feat)
                        feat_data = jacc_data[n_feat]
                        
                        # Extract each method pair
                        for pair in method_pairs
                            if haskey(feat_data, pair)
                                value = feat_data[pair]
                                push!(jaccard_all_datasets[dataset][pair][n_feat], value)
                            end
                        end
                    end
                end
            end
        end
        
        # Print summary for this dataset
        if verbose
            for pair in method_pairs
                all_vals = Float64[]
                for n_feat in top_n_feats_candidates
                    append!(all_vals, jaccard_all_datasets[dataset][pair][n_feat])
                end
                
                if !isempty(all_vals)
                    println("  $pair: mean = $(round(mean(all_vals), digits=3)), " *
                            "std = $(round(std(all_vals), digits=3))")
                end
            end
            println()
        end
    end
    
    verbose && println("✓ Jaccard indices extracted for all datasets!")
    
    return jaccard_all_datasets
end
# ============================================================================
# Helper function to compute mean curve
    function mean_curve(jaccard_results, n_features_list; method_pair::String="mRMR_Boruta")
        haskey(jaccard_results, method_pair) || return Int[], Float64[]
        xs = Int[]
        ys = Float64[]
        for n in sort(collect(n_features_list))
            if haskey(jaccard_results[method_pair], n) && !isempty(jaccard_results[method_pair][n])
                push!(xs, n)
                push!(ys, mean(jaccard_results[method_pair][n]))
            end
        end
        return xs, ys
    end
# ============================================================================
#Plot Jaccard index comparison across datasets for a specific method pair.
#
# Arguments
#   - `jaccard_all_datasets`: Dictionary with Jaccard indices per dataset
#   - `top_n_feats_candidates`: Vector of feature counts to plot
#   - `dataset_labels`: Dictionary mapping dataset names to display labels
#   - `line_colors`: Dictionary mapping dataset names to colors
#   - `method_pair`: Which method pair to plot (default: "mRMR_Boruta")
#   - `title`: Plot title
#   - `ylim`: Y-axis limits
#   - `size`: Plot size as (width, height) tuple
#   - `dpi`: Resolution for saving
#
# Returns
#   - Plots.Plot object
# ============================================================================
function plot_jaccard_comparison(jaccard_all_datasets, top_n_feats_candidates, 
                                 dataset_labels, line_colors;
                                 method_pair::String="mRMR_Boruta",
                                 title = "",
                                 ylim=(0, 0.61),
                                 size=(950, 600),
                                 dpi=300)

# Ensure legend label shows "male-only" instead of "male"
    dataset_labels = copy(dataset_labels)
    dataset_labels["male"] = "Male-only"
    dataset_labels["female"] = "Female-only"

    # Initialize plot
    p = Plots.plot(
        xlabel = "Number of Features",
        ylabel = "Jaccard Index",
        title  = title,
        legend = :bottomright,
        ylim   = ylim,
        left_margin = 10mm,
        right_margin = 10mm,
        bottom_margin = 10mm,
        titlefontsize = 12,
        guidefontsize  = 14,  
        tickfontsize=12,
        legendfontsize = 12, 
        size = size,
        dpi = dpi
    )
    
    # Plot each dataset
    for ds in ["baseline", "sex aware", "male", "female"]
        haskey(jaccard_all_datasets, ds) || continue
        xs, ys = mean_curve(jaccard_all_datasets[ds], top_n_feats_candidates;method_pair)
        isempty(xs) && continue
        
        Plots.plot!(p, xs, ys;
            label     = dataset_labels[ds],
            color     = line_colors[ds],
            linestyle = :solid,
            linewidth = 4
        )
    end
    
    Plots.plot!(; framestyle=:box)
    
    return p
end
# ========================================================
# Create performance plots showing top performing classifier-method combinations for each dataset.
#
# Arguments
#
#- `all_results`: Dictionary containing nested CV results for all datasets
#- `dataset_order`: Vector of dataset names to process
#- `top_n_feats_candidates`: Vector of feature counts to evaluate
#- `n_outer_folds`: Number of outer CV folds
#- `classifiers`: Vector of classifier names
#- `classifier_display_names`: Dictionary mapping classifier names to display labels
#- `classifier_colors`: Dictionary mapping classifier names to colors
#- `method_markers`: Dictionary mapping feature selection methods to marker styles
#- `methods`: Tuple of feature selection methods to compare (default: ("mRMR", "Boruta"))
#- `n_top`: Number of top combinations to plot (default: 3)
#- `ylim`: Y-axis limits (default: (0.0, 0.7))
#- `filter_sex_aware_rf`: Whether to filter Random Forest from Sex-Aware dataset (default: true)
#- `verbose`: Whether to print ranking information (default: true)
#
# Returns
#- Vector of Plots.Plot objects, one per dataset
# ========================================================
function plot_top_performers(all_results, dataset_order, top_n_feats_candidates, 
                             n_outer_folds, classifiers,
                             classifier_display_names, classifier_colors, method_markers;
                             methods=("mRMR", "Boruta"),
                             n_top=3,
                             ylim=(0.0, 0.7),
                             filter_sex_aware_rf=true,
                             verbose=true)
    
    all_plots = []
    
    for dataset_name in dataset_order
        nested_cv_results = all_results[dataset_name]
        
        # Calculate MCC for all combinations and find max for each
        mcc_data = Dict()
        max_mcc_ranking = []
        
        for method in methods
            mcc_data[method] = Dict()
            
            for classifier in classifiers
                mcc_data[method][classifier] = []
                
                for n in sort(collect(top_n_feats_candidates))
                    mcc_values = [nested_cv_results[fold_idx][method]["$(n)_features"][classifier].mcc 
                                 for fold_idx in 1:n_outer_folds]
                    
                    push!(mcc_data[method][classifier], mean(mcc_values))
                end
                
                # Store max MCC for ranking
                max_mcc = maximum(mcc_data[method][classifier])
                push!(max_mcc_ranking, (method, classifier, max_mcc))
            end
        end
        
        # Sort by max MCC
        sort!(max_mcc_ranking, by = x -> x[3], rev = true)
        
        dataset_label = dataset_name == "baseline" ? "Full Dataset" : dataset_name
        
        # If this is the Sex Aware plot, optionally filter RF before taking top N
        is_sex_aware = lowercase(replace(dataset_label, "_" => " ")) == "sex aware"
        
        if is_sex_aware && filter_sex_aware_rf
            filtered = filter(x -> !(
                (x[1] == "Boruta" && x[2] == "RandomForest") ||
                (x[1] == "mRMR"   && x[2] == "RandomForest")
            ), max_mcc_ranking)
            top_n_combinations = filtered[1:min(n_top, length(filtered))]
        else
            top_n_combinations = max_mcc_ranking[1:min(n_top, length(max_mcc_ranking))]
        end
        
        dataset_key = lowercase(replace(dataset_name, "_" => " ", "-" => " "))
        
        dataset_label = dataset_key == "baseline"  ? "Full Dataset" :
                        dataset_key == "sex aware" ? "Sex-Aware" :
                        dataset_name
        
        # Print ranking if verbose
        if verbose
            println("\n" * "="^70)
            println("DATASET: $(uppercase(dataset_label))")
            println("="^70)
            println("Top $(n_top) Classifier-Method Combinations:")
            for (i, (method, classifier, max_mcc)) in enumerate(top_n_combinations)
                println("$i. $(classifier_display_names[classifier]) ($method): MCC = $(round(max_mcc, digits=3))")
            end
        end
        
        feature_counts = sort(collect(top_n_feats_candidates))
        
        # Create plot for this dataset
        p = Plots.plot(
            xlabel="Number of Features",
            ylabel="MCC",
            title="$(titlecase(dataset_label))",
            legend=:bottomright,
            framestyle=:box,
            grid=true,
            gridalpha=0.3,
            ylim=ylim,
            xticks=10:10:100,
            titlefontsize  = 13,
            guidefontsize  = 12,
            tickfontsize   = 11,
            legendfontsize = 11,  
            left_margin   = 8mm,
            bottom_margin = 8mm,
            right_margin  = 6mm,
            top_margin    = 8mm,

        )
        
        # Plot top N combinations
        for (method, classifier, _) in top_n_combinations
            Plots.plot!(p,
                feature_counts,
                mcc_data[method][classifier],
                label="$(classifier_display_names[classifier]) ($method)",
                linewidth=2.5,
                linestyle=:solid,
                marker=method_markers[method],
                markersize=5,
                markerstrokewidth=0,
                color=classifier_colors[classifier]
            )
        end
        
        push!(all_plots, p)
    end
    
    return all_plots
end
# =====================================
# confunsion matrix
# ======================================
 function to_int(x)
        ismissing(x) && return missing
        s = strip(string(x))
        isempty(s) && return missing
        try
            return parse(Int, s)
        catch
            return Int(round(parse(Float64, s)))  # handles "100.0"
        end
end

# ============================================================================
# generate_table2: Table 2 — inner cross-validation performance (95% CI)
#   inner_df      DataFrame from inner_fold_metrics CSV
#                 required columns: Dataset, Method, N_Features, Classifier,
#                                   Split, MCC, F1, Accuracy, BACC,
#                                   Sensitivity, Specificity
#   best_configs  Dict: dataset_key => (method=..., n_features=..., classifier=...)
#   Filters Split == "val", aggregates 15 inner-fold observations per config.
#   CI format: "mean (lower–upper)" using t(df=14, α=0.025).
#   Prints via PrettyTables and returns a DataFrame.
# ============================================================================
function generate_table2(inner_df::DataFrame, best_configs::Dict)
    display_labels = Dict(
        "baseline"  => "Full Dataset",
        "sex aware" => "Sex-Aware",
        "male"      => "Male-Specific",
        "female"    => "Female-Specific"
    )
    dataset_order = ["baseline", "sex aware", "male", "female"]
    t_crit = 2.1448  # t(df=14, α=0.025, two-tailed) for n=15 observations

    function ci95(v)
        m  = mean(v)
        se = std(v) / sqrt(length(v))
        @sprintf("%.3f (%.3f–%.3f)", m, m - t_crit * se, m + t_crit * se)
    end

    model_config = String[]; fs_method = String[]; n_feats = Int[]
    classif      = String[]; mcc_ci    = String[]; f1_ci   = String[]
    acc_ci       = String[]; bacc_ci   = String[]; sens_ci = String[]; spec_ci = String[]

    for ds in dataset_order
        cfg = best_configs[ds]
        sub = filter(r -> r.Dataset    == ds             &&
                          r.Method     == cfg.method     &&
                          r.N_Features == cfg.n_features &&
                          r.Classifier == cfg.classifier &&
                          r.Split      == "val", inner_df)
        if nrow(sub) < 2
            @warn "generate_table2: fewer than 2 val rows for dataset=$ds" *
                  " ($(cfg.method), N=$(cfg.n_features), $(cfg.classifier))"
            continue
        end
        push!(model_config, display_labels[ds])
        push!(fs_method,    cfg.method)
        push!(n_feats,      cfg.n_features)
        push!(classif,      cfg.classifier)
        push!(mcc_ci,       ci95(sub.MCC))
        push!(f1_ci,        ci95(sub.F1))
        push!(acc_ci,       ci95(sub.Accuracy))
        push!(bacc_ci,      ci95(sub.BACC))
        push!(sens_ci,      ci95(sub.Sensitivity))
        push!(spec_ci,      ci95(sub.Specificity))
    end

    tbl = DataFrame(
        "Model Configuration"      => model_config,
        "Feature Selection Method" => fs_method,
        "Number of Features"       => n_feats,
        "Classifier"               => classif,
        "MCC"                      => mcc_ci,
        "F1"                       => f1_ci,
        "Accuracy"                 => acc_ci,
        "Balanced Accuracy"        => bacc_ci,
        "Sensitivity"              => sens_ci,
        "Specificity"              => spec_ci
    )
    pretty_table(tbl; header_crayon = crayon"bold")
    return tbl
end

# ============================================================================
# plot_inner_vs_outer_mcc: Figure 6 — inner train / inner val / outer test MCC
#   Shows the nested CV hierarchy for one dataset × method × N configuration.
#   inner_df     DataFrame from inner_fold_metrics CSV
#   dataset_name one of "baseline", "sex aware", "male", "female"
#   method       feature selection method string
#   n_features   N matching the best Table 1 config for this dataset
#   clf_colors   Dict mapping classifier key to color (hex string or symbol)
# Returns CairoMakie.Figure
# ============================================================================
function plot_inner_vs_outer_mcc(
    inner_df::DataFrame,
    dataset_name::String,
    method::String,
    n_features::Int;
    classifiers = [
        "RandomForest", "RandomForest_TUNED",
        "NeuralNetwork", "NeuralNetwork_TUNED",
        "NaiveBayes", "LogisticRegression",
        "AdaBoost", "AdaBoost_TUNED",
        "SVM_RBF", "SVM_RBF_TUNED"
    ],
    clf_display = Dict(
        "RandomForest"        => "RF",       "RandomForest_TUNED"  => "RF (T)",
        "NeuralNetwork"       => "NN",       "NeuralNetwork_TUNED" => "NN (T)",
        "NaiveBayes"          => "NB",       "LogisticRegression"  => "LR",
        "AdaBoost"            => "AB",       "AdaBoost_TUNED"      => "AB (T)",
        "SVM_RBF"             => "SVM",      "SVM_RBF_TUNED"       => "SVM (T)"
    ),
    clf_colors = Dict(
        "RandomForest"        => "#dc267f",  "RandomForest_TUNED"  => "#dc267f",
        "NeuralNetwork"       => "#785ef0",  "NeuralNetwork_TUNED" => "#785ef0",
        "NaiveBayes"          => "#2ca02c",  "LogisticRegression"  => "#9467bd",
        "AdaBoost"            => "#ffb000",  "AdaBoost_TUNED"      => "#ffb000",
        "SVM_RBF"             => "#fe6100",  "SVM_RBF_TUNED"       => "#fe6100"
    )
)
    t_crit_test = 2.7764  # t(df=4, α=0.025, two-tailed) for 5 outer folds
    _ds_labels  = Dict(
        "baseline"  => "Full Dataset", "sex aware" => "Sex-Aware",
        "male"      => "Male-Specific", "female"    => "Female-Specific"
    )

    sub     = filter(r -> r.Dataset    == dataset_name &&
                          r.Method     == method       &&
                          r.N_Features == n_features,   inner_df)
    n_clf   = length(classifiers)
    clf_tck = [get(clf_display, c, c) for c in classifiers]

    train_x = Float64[]; train_y = Float64[]
    val_x   = Float64[]; val_y   = Float64[]

    fig = CairoMakie.Figure(size = (max(900, 110 * n_clf), 560))
    ax  = CairoMakie.Axis(fig[1, 1];
        title              = "$(get(_ds_labels, dataset_name, dataset_name))" *
                             "  |  $method, N=$n_features",
        xlabel             = "Classifier",
        ylabel             = "MCC",
        xticks             = (1:n_clf, clf_tck),
        xticklabelrotation = π / 5,
        titlesize          = 14,
        xlabelsize         = 13,
        ylabelsize         = 13
    )

    for (i, clf) in enumerate(classifiers)
        c_sub = filter(r -> r.Classifier == clf, sub)
        trn   = filter(r -> r.Split == "train", c_sub).MCC
        vl    = filter(r -> r.Split == "val",   c_sub).MCC
        isempty(trn) || begin
            append!(train_x, fill(i - 0.28, length(trn)))
            append!(train_y, trn)
        end
        isempty(vl) || begin
            append!(val_x, fill(float(i), length(vl)))
            append!(val_y, vl)
        end
    end

    isempty(train_x) || CairoMakie.boxplot!(ax, train_x, train_y;
        color = (:gray50, 0.60), label = "Inner Train", width = 0.22)
    isempty(val_x) || CairoMakie.boxplot!(ax, val_x, val_y;
        color = (:steelblue, 0.75), label = "Inner Val",   width = 0.22)

    # Legend placeholder for test (avoids duplicate entries from per-classifier loop)
    CairoMakie.scatter!(ax, Float64[], Float64[];
        color = :black, markersize = 10, label = "Outer Test (95% CI)")

    for (i, clf) in enumerate(classifiers)
        c_sub = filter(r -> r.Classifier == clf && r.Split == "test", sub)
        isempty(c_sub) && continue
        tst = c_sub.MCC
        m   = mean(tst)
        se  = length(tst) > 1 ? std(tst) / sqrt(length(tst)) : 0.0
        lo  = m - t_crit_test * se
        hi  = m + t_crit_test * se
        CairoMakie.rangebars!(ax, [i + 0.28], [lo], [hi];
            color = :black, linewidth = 2)
        CairoMakie.scatter!(ax, [i + 0.28], [m];
            color = :black, markersize = 10)
    end

    CairoMakie.axislegend(ax; position = :rt, framevisible = true)
    return fig
end

# ============================================================================
# plot_nn_loss_curves: Figure 7 — NN epoch-level training loss (2×3 panels)
#   nn_losses    Dict keyed [dataset][outer_fold_idx][method][feat_key]
#   dataset_name one of "baseline", "sex aware", "male", "female"
#   method       feature selection method used in the best NN run
#   feat_key     feature-count key in the dict, e.g. "100_features"
# Returns CairoMakie.Figure (2×3 grid: 5 per-fold panels + 1 aggregate)
# ============================================================================
function plot_nn_loss_curves(
    nn_losses::Dict,
    dataset_name::String,
    method::String,
    feat_key::String
)
    _ds_labels  = Dict(
        "baseline"  => "Full Dataset", "sex aware" => "Sex-Aware",
        "male"      => "Male-Specific", "female"    => "Female-Specific"
    )
    n_folds     = 5
    fold_colors = ["#785ef0", "#648fff", "#dc267f", "#ffb000", "#fe6100"]

    fold_losses = Vector{Float64}[]
    for fi in 1:n_folds
        try
            entry = nn_losses[dataset_name][fi][method][feat_key]
            lv = if entry isa AbstractVector
                entry
            elseif entry isa AbstractDict
                if haskey(entry, "NeuralNetwork_TUNED_losses") && !isempty(entry["NeuralNetwork_TUNED_losses"])
                    entry["NeuralNetwork_TUNED_losses"]
                else
                    entry["NeuralNetwork_losses"]
                end
            else
                error("unexpected type $(typeof(entry))")
            end
            push!(fold_losses, Float64.(lv))
        catch
            @warn "plot_nn_loss_curves: missing key for " *
                  "$dataset_name fold $fi $method $feat_key"
            push!(fold_losses, Float64[])
        end
    end

    fig = CairoMakie.Figure(size = (1400, 850))
    CairoMakie.Label(fig[0, :],
        "$(get(_ds_labels, dataset_name, dataset_name)) — NN Training Loss  |  " *
        "$method, $feat_key";
        fontsize = 15, font = :bold)

    positions = [(1,1),(1,2),(1,3),(2,1),(2,2)]
    for (k, (r, c)) in enumerate(positions)
        ax = CairoMakie.Axis(fig[r, c];
            title     = "Outer Fold $k",
            xlabel    = r == 2 ? "Epoch" : "",
            ylabel    = c == 1 ? "Training Loss" : "",
            titlesize = 13)
        lv = fold_losses[k]
        isempty(lv) && continue
        CairoMakie.lines!(ax, 1:length(lv), lv;
            color = fold_colors[k], linewidth = 2)
    end

    ax_agg = CairoMakie.Axis(fig[2, 3];
        title     = "Aggregate (mean ± 1 SD)",
        xlabel    = "Epoch",
        titlesize = 13)
    valid = filter(!isempty, fold_losses)
    if length(valid) >= 2
        min_len = minimum(length.(valid))
        mat     = hcat([v[1:min_len] for v in valid]...)
        μ       = vec(mean(mat, dims = 2))
        σ       = vec(std(mat,  dims = 2))
        ep      = 1:min_len
        CairoMakie.band!(ax_agg, ep, μ .- σ, μ .+ σ; color = (:gray60, 0.35))
        CairoMakie.lines!(ax_agg, ep, μ; color = :black, linewidth = 2)
    end

    CairoMakie.colgap!(fig.layout, 10)
    CairoMakie.rowgap!(fig.layout, 10)
    return fig
end
