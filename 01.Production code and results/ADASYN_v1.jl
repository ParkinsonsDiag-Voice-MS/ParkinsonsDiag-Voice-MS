# =========================================================================
# ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning (Optimized)
#
# Implementation of the ADASYN algorithm from:
# He, H., Bai, Y., Garcia, E. A., & Li, S. (2008). 
# "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning"
# IEEE International Joint Conference on Neural Networks (IJCNN 2008).
#
# ADASYN adaptively generates synthetic minority class samples according to their
# distribution, creating more synthetic data for harder-to-learn minority samples.
#
# Optimizations in this version:
# 	1. **Efficient k-NN search**: Uses NearestNeighbors.jl (KDTree/BallTree) instead of brute-force
# 	2. **Type generality**: Preserves element type of input matrices
# 	3. **Parallelization**: Thread-based parallelization for synthetic sample generation
#
# Key Features
# 	- Reduces bias from class imbalance
# 	- Adaptively shifts decision boundary toward difficult examples
# 	- Uses density distribution to weight minority sample importance
# 	- Scales efficiently to larger datasets with optimized k-NN search
# 	- Utilizes multicore processors for faster generation
#----------------------------------------------------------------------------
# Basic Usage
# ```julia
# using Random
# include("ADASYN.jl")
# using .ADASYN
# ```
#
# Generate imbalanced data
# 	X = randn(100, 10)
# 	y = vcat(zeros(Int, 80), ones(Int, 20))
#
# Apply ADASYN (automatically uses available threads)
# 	X_resampled, y_resampled = adasyn_fit_resample(X, y; β=1.0, K=5,
#                                                  rng=MersenneTwister(42))
#--------------------------------------------------------------------------

module ADASYN

using Random
using Statistics
using LinearAlgebra
using NearestNeighbors  # OPTIMIZATION 1: Efficient k-NN search

export adasyn_fit_resample, adasyn

# =========================================================================	
#    adasyn_fit_resample(X, y; β=1.0, d_th=0.75, K=5, rng=Random.GLOBAL_RNG)
#
# Apply ADASYN (Adaptive Synthetic Sampling) to balance an imbalanced dataset.
#
# 	Arguments
# 		- `X::AbstractMatrix{<:Real}`: Feature matrix (samples × features)
# 		- `y::AbstractVector`: Class labels (binary: 0/1, -1/1, true/false, etc.)
# 		- `β::Real=1.0`: Balance level parameter ∈ [0,1]. β=1 creates fully balanced dataset
# 		- `d_th::Real=0.75`: Maximum tolerated imbalance ratio. ADASYN only applied if d < d_th
# 		- `K::Integer=5`: Number of nearest neighbors to consider
# 		- `rng::AbstractRNG=Random.GLOBAL_RNG`: Random number generator for reproducibility
#
# Returns
# 		- `X_resampled::Matrix`: Resampled feature matrix (type-preserving)
# 		- `y_resampled::Vector`: Resampled labels
#
# Algorithm
# 	1. Compute imbalance degree d = ms/ml (minority/majority ratio)
# 	2. If d < d_th:
#   	- Calculate G = (ml - ms) × β synthetic samples to generate
#  		- For each minority sample xi:
#     		* Find K nearest neighbors (using KDTree for efficiency)
#     		* Compute ri = Δi/K where Δi = # majority neighbors
#     		* Normalize: r̂i = ri / Σri (density distribution)
#     		* Generate gi = r̂i × G synthetic samples
#  		- For each synthetic sample (parallelized):
#    		* Choose random minority neighbor xz from K nearest neighbors
#     		* Generate: s = xi + λ(xz - xi) where λ ~ Uniform(0,1)
#
# 	Edge Cases
# 		- If d ≥ d_th: Returns original data (already balanced enough)
# 		- If minority class has < K+1 samples: Reduces K automatically
# 		- If all ri = 0 (all minority samples surrounded by minority): Falls back to uniform generation
#
# Performance Notes
#	- Uses KDTree for O(log n) neighbor queries vs O(n) brute-force
#	- Parallelizes synthetic sample generation across available threads
#	- Preserves input element type for memory efficiency
#
# Example
# 	```julia
# 		X = randn(100, 5)
# 		y = vcat(zeros(Int, 85), ones(Int, 15))  # 85 majority, 15 minority
# 		X_res, y_res = adasyn_fit_resample(X, y; β=1.0, K=5)
#       @assert sum(y_res .== 1) > sum(y .== 1)  # More minority samples
#	```
#
# Reference
# ADASYN implementation based on:
#   (1) He, H., Bai, Y., Garcia, E. A., & Li, S. (2008). 
#       "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning."
#       IEEE IJCNN, pp. 1322–1328.
#
#   (2) Python reference implementation:
#       https://github.com/stavskal/ADASYN
#
#   (3) Algorithmic refinements and optimization guidance supported by 
#       large language model (LLM) analysis using OpenAI’s ChatGPT AND Claude.
# =========================================================================	

function adasyn_fit_resample(
    X::AbstractMatrix{<:Real},
    y::AbstractVector;
    β::Real = 1.0,
    d_th::Real = 0.75,
    K::Integer = 5,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    # Validate inputs
    size(X, 1) == length(y) || throw(DimensionMismatch("X and y must have same number of samples"))
    0 <= β <= 1 || throw(ArgumentError("β must be in [0, 1]"))
    0 < d_th <= 1 || throw(ArgumentError("d_th must be in (0, 1]"))
    K > 0 || throw(ArgumentError("K must be positive"))
    
    # Ensure y is a vector of comparable elements
    y_vec = vec(y)
    
    # Identify minority and majority classes
    unique_classes = unique(y_vec)
    length(unique_classes) == 2 || throw(ArgumentError("ADASYN requires exactly 2 classes"))
    
    # Count class sizes
    class_counts = Dict(c => count(==(c), y_vec) for c in unique_classes)
    minority_class = argmin(class_counts)
    majority_class = argmax(class_counts)
    
    ms = class_counts[minority_class]  # Number of minority samples
    ml = class_counts[majority_class]  # Number of majority samples
    
    # Calculate imbalance degree
    d = ms / ml
    
    # If already balanced enough, return original data
    # OPTIMIZATION 2: Type-preserving conversion
    if d >= d_th
        return Matrix{eltype(X)}(X), collect(y_vec)
    end
    
    # Check if we have enough minority samples
    if ms < 2
        @warn "Too few minority samples ($ms) for ADASYN. Returning original data."
        return Matrix{eltype(X)}(X), collect(y_vec)
    end
    
    # Adjust K if necessary
    K_actual = min(K, ms - 1)
    if K_actual < K
        @warn "Reduced K from $K to $K_actual (insufficient minority samples)"
    end
    
    # Extract minority and majority samples
    minority_mask = y_vec .== minority_class
    majority_mask = y_vec .== majority_class
    
    X_minority = X[minority_mask, :]
    X_majority = X[majority_mask, :]
    
    # Generate synthetic samples
    X_synthetic = adasyn(X_minority, X_majority; β=β, K=K_actual, G_total=nothing, rng=rng)
    
    # Combine original and synthetic data
    X_resampled = vcat(X, X_synthetic)
    y_synthetic = fill(minority_class, size(X_synthetic, 1))
    y_resampled = vcat(y_vec, y_synthetic)
    
    return X_resampled, y_resampled
end


"""
    adasyn(X_minority, X_majority; β=1.0, K=5, G_total=nothing, rng=Random.GLOBAL_RNG)

Low-level ADASYN function that generates synthetic minority samples.

# Arguments
- `X_minority::AbstractMatrix{<:Real}`: Minority class samples (samples × features)
- `X_majority::AbstractMatrix{<:Real}`: Majority class samples (samples × features)
- `β::Real=1.0`: Balance level parameter ∈ [0,1]
- `K::Integer=5`: Number of nearest neighbors
- `G_total::Union{Integer,Nothing}=nothing`: Total synthetic samples to generate. If nothing, uses G = (ml - ms) × β
- `rng::AbstractRNG=Random.GLOBAL_RNG`: Random number generator

# Returns
- `X_synthetic::Matrix`: Generated synthetic minority samples (G × features)

# Algorithm Details
For each minority sample xi:
1. Find K nearest neighbors in full feature space (using KDTree)
2. Compute ri = Δi/K where Δi = number of majority neighbors
3. Normalize to density distribution: r̂i = ri / Σri
4. Generate gi = round(r̂i × G) synthetic samples
5. For each synthetic sample (parallelized across threads):
   - Randomly select minority neighbor xz from K nearest neighbors
   - Generate: s = xi + λ(xz - xi) where λ ~ Uniform(0,1)
"""
function adasyn(
    X_minority::AbstractMatrix{<:Real},
    X_majority::AbstractMatrix{<:Real};
    β::Real = 1.0,
    K::Integer = 5,
    G_total::Union{Integer,Nothing} = nothing,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    ms, n_features = size(X_minority)
    ml = size(X_majority, 1)
    
    # Validate dimensions
    size(X_majority, 2) == n_features || throw(DimensionMismatch("X_minority and X_majority must have same number of features"))
    
    # Calculate total synthetic samples to generate
    if G_total === nothing
        G = round(Int, (ml - ms) * β)
    else
        G = G_total
    end
    
    if G <= 0
        # OPTIMIZATION 2: Type-preserving empty matrix
        return Matrix{eltype(X_minority)}(undef, 0, n_features)
    end
    
    # Combine minority and majority samples for nearest neighbor search
    X_combined = vcat(X_minority, X_majority)
    n_total = ms + ml
    
    # OPTIMIZATION 1: Build KDTree for efficient k-NN queries
    # Transpose because NearestNeighbors expects features × samples
    kdtree = KDTree(Matrix(X_combined'))
    
    # Pre-allocate arrays
    r = zeros(Float64, ms)
    
    # For each minority sample, find K nearest neighbors and compute ri
    for i in 1:ms
        xi = X_minority[i, :]
        
        # OPTIMIZATION 1: Efficient k-NN query using KDTree
        # Find K+1 neighbors to account for the point itself
        idxs, dists = knn(kdtree, xi, K + 1, true)  # true = sort by distance
        
        # Filter out the point itself (distance ≈ 0) and get K neighbors
        knn_indices = Int[]
        for (idx, dist) in zip(idxs, dists)
            # Skip if it's the same sample (distance very small)
            # Note: idx is in combined space, i is in minority space
            if idx == i && dist < 1e-10
                continue
            end
            push!(knn_indices, idx)
            if length(knn_indices) >= K
                break
            end
        end
        
        # Count how many neighbors belong to majority class
        # Majority class samples are indexed from ms+1 to n_total in X_combined
        Δi = count(idx -> idx > ms, knn_indices)
        
        # Compute ratio
        r[i] = Δi / K
    end
    
    # Normalize to create density distribution
    sum_r = sum(r)
    
    # Handle edge case: all minority samples surrounded by minority neighbors
    if sum_r ≈ 0.0
        @warn "All minority samples surrounded by minority neighbors. Using uniform distribution."
        r_hat = fill(1.0 / ms, ms)
    else
        r_hat = r ./ sum_r
    end
    
    # Calculate number of synthetic samples for each minority sample
    g = zeros(Int, ms)
    g_float = r_hat .* G
    
    # Round and ensure sum equals G (handle rounding errors)
    for i in 1:ms
        g[i] = round(Int, g_float[i])
    end
    
    # Adjust for rounding errors
    diff = G - sum(g)
    if diff > 0
        # Add remaining samples to highest r_hat values
        sorted_indices = sortperm(r_hat, rev=true)
        for i in 1:diff
            g[sorted_indices[i]] += 1
        end
    elseif diff < 0
        # Remove excess samples from highest g values
        sorted_indices = sortperm(g, rev=true)
        for i in 1:(-diff)
            if g[sorted_indices[i]] > 0
                g[sorted_indices[i]] -= 1
            end
        end
    end
    
    # OPTIMIZATION 1: Build KDTree for minority samples k-NN queries
    kdtree_minority = KDTree(Matrix(X_minority'))
    
    # Pre-compute K nearest minority neighbors for each minority sample
    # This avoids redundant k-NN queries in the parallel loop
    minority_neighbors_cache = Vector{Vector{Int}}(undef, ms)
    for i in 1:ms
        xi = X_minority[i, :]
        
        # OPTIMIZATION 1: Efficient k-NN query on minority samples only
        idxs, dists = knn(kdtree_minority, xi, K + 1, true)
        
        # Get K nearest minority neighbors (excluding self)
        neighbors = Int[]
        for (idx, dist) in zip(idxs, dists)
            if idx != i || dist > 1e-10
                push!(neighbors, idx)
            end
            if length(neighbors) >= K
                break
            end
        end
        
        # Handle case where we have fewer neighbors than K
        if isempty(neighbors)
            neighbors = [i]  # Use the sample itself if no neighbors
        end
        
        minority_neighbors_cache[i] = neighbors
    end
    
    # OPTIMIZATION 2: Type-preserving matrix allocation
    X_synthetic = Matrix{eltype(X_minority)}(undef, G, n_features)
    
    # Create deterministic per-sample RNG states for reproducibility
    # Generate all random states before parallelization
    rng_seeds = [rand(rng, UInt32) for _ in 1:ms]
    random_values = Dict{Int, Vector{Tuple{Int, eltype(X_minority)}}}()
    
    # Pre-generate random values for deterministic parallel execution
    for i in 1:ms
        if g[i] > 0
            # Create a temporary RNG with the saved state for this minority sample
            temp_rng = MersenneTwister(rng_seeds[i])
            
            # Generate g[i] pairs of (neighbor_index, λ)
            vals = Tuple{Int, eltype(X_minority)}[]
            for _ in 1:g[i]
                zi_idx = rand(temp_rng, minority_neighbors_cache[i])
                λ = rand(temp_rng, eltype(X_minority))
                push!(vals, (zi_idx, λ))
            end
            random_values[i] = vals
        end
    end
    
    # Calculate cumulative starting indices for each minority sample
    cumsum_g = cumsum(g)
    start_indices = vcat([1], cumsum_g[1:end-1] .+ 1)
    
    # OPTIMIZATION 3: Parallel generation of synthetic samples
    # Use Threads.@threads for multicore performance
    Threads.@threads for i in 1:ms
        if g[i] == 0
            continue
        end
        
        xi = X_minority[i, :]
        neighbors = minority_neighbors_cache[i]
        rand_vals = random_values[i]
        
        # Starting index in X_synthetic for this minority sample
        start_idx = start_indices[i]
        
        # Generate g[i] synthetic samples for this minority point
        for j in 1:g[i]
            # Use pre-generated random values for reproducibility
            zi_idx, λ = rand_vals[j]
            xzi = X_minority[zi_idx, :]
            
            # Generate synthetic sample: si = xi + λ(xzi - xi)
            si = xi .+ λ .* (xzi .- xi)
            
            # Store in the correct position
            X_synthetic[start_idx + j - 1, :] = si
        end
    end
    
    return X_synthetic
end


# ============================================================================
# Utility Functions (not exported)
# ============================================================================

"""
    check_adasyn(X, y, X_res, y_res)

Helper function to validate ADASYN output.
"""
function check_adasyn(X, y, X_res, y_res)
    # Check dimensions
    @assert size(X_res, 2) == size(X, 2) "Feature dimensions preserved"
    @assert size(X_res, 1) == length(y_res) "Consistent resampled dimensions"
    
    # Check minority class increased
    minority_class = argmin(Dict(c => count(==(c), y) for c in unique(y)))
    original_minority = count(==(minority_class), y)
    resampled_minority = count(==(minority_class), y_res)
    @assert resampled_minority >= original_minority "Minority class count increased"
    
    println("✓ ADASYN validation passed")
    println("  Original: $(length(y)) samples ($(original_minority) minority)")
    println("  Resampled: $(length(y_res)) samples ($(resampled_minority) minority)")
end


# ============================================================================
# Basic Tests
# ============================================================================

"""
    test_adasyn()

Run basic tests for ADASYN implementation.
"""
function test_adasyn()
    println("\n" * "="^60)
    println("Testing Optimized ADASYN Implementation")
    println("="^60)
    println("Number of threads: ", Threads.nthreads())
    
    # Test 1: Basic functionality
    println("\nTest 1: Basic functionality")
    rng = MersenneTwister(42)
    X = randn(rng, 100, 10)
    y = vcat(zeros(Int, 80), ones(Int, 20))
    
    X_res, y_res = adasyn_fit_resample(X, y; β=1.0, K=5, rng=rng)
    check_adasyn(X, y, X_res, y_res)
    
    # Test 2: Different label encoding
    println("\nTest 2: Different label encoding (-1/1)")
    y2 = vcat(fill(-1, 80), fill(1, 20))
    X_res2, y_res2 = adasyn_fit_resample(X, y2; β=0.5, K=5, rng=MersenneTwister(42))
    check_adasyn(X, y2, X_res2, y_res2)
    
    # Test 3: Already balanced data
    println("\nTest 3: Already balanced data (d ≥ d_th)")
    y3 = vcat(zeros(Int, 50), ones(Int, 50))
    X3 = randn(MersenneTwister(42), 100, 10)
    X_res3, y_res3 = adasyn_fit_resample(X3, y3; β=1.0, d_th=0.75, K=5, rng=MersenneTwister(42))
    @assert size(X_res3) == size(X3) "Balanced data unchanged"
    println("  ✓ Balanced data returned unchanged")
    
    # Test 4: Deterministic with fixed RNG
    println("\nTest 4: Deterministic behavior with fixed RNG")
    rng1 = MersenneTwister(123)
    rng2 = MersenneTwister(123)
    X_res4a, y_res4a = adasyn_fit_resample(X, y; β=1.0, K=5, rng=rng1)
    X_res4b, y_res4b = adasyn_fit_resample(X, y; β=1.0, K=5, rng=rng2)
    @assert X_res4a ≈ X_res4b "Deterministic with same seed"
    @assert y_res4a == y_res4b "Deterministic labels"
    println("  ✓ Deterministic behavior confirmed (even with parallelization)")
    
    # Test 5: Small K
    println("\nTest 5: Small K value")
    X_res5, y_res5 = adasyn_fit_resample(X, y; β=1.0, K=2, rng=MersenneTwister(42))
    check_adasyn(X, y, X_res5, y_res5)
    
    # Test 6: Partial balance (β < 1)
    println("\nTest 6: Partial balance (β=0.3)")
    X_res6, y_res6 = adasyn_fit_resample(X, y; β=0.3, K=5, rng=MersenneTwister(42))
    minority_count = count(==(1), y_res6)
    majority_count = count(==(0), y_res6)
    println("  Minority: $minority_count, Majority: $majority_count")
    @assert minority_count > 20 && minority_count < 80 "Partial balance achieved"
    
    # Test 7: Type preservation
    println("\nTest 7: Type preservation")
    X_f32 = Float32.(X)
    X_res7, y_res7 = adasyn_fit_resample(X_f32, y; β=1.0, K=5, rng=MersenneTwister(42))
    @assert eltype(X_res7) == Float32 "Float32 type preserved"
    println("  ✓ Input type Float32 preserved in output")
    
    # Test 8: Performance comparison (optional, commented out for regular tests)
    # println("\nTest 8: Performance comparison")
    # using BenchmarkTools
    # println("  Benchmarking with K=5, 100 samples, 10 features:")
    # @btime adasyn_fit_resample($X, $y; β=1.0, K=5, rng=MersenneTwister(42))
    
    println("\n" * "="^60)
    println("All tests passed! ✓")
    println("="^60 * "\n")
end


# Run tests if module is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_adasyn()
end

end # module ADASYN
