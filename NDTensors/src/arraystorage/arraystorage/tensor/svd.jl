# TODO: Rewrite this function to be more modern:
# 1. List keyword arguments in function signature.
# 2. Output `Spectrum` as a keyword argument that gets overwritten.
# 3. Dispatch on `alg`.
# 4. Remove keyword argument deprecations.
# 5. Make this into two layers, one that handles indices and one that works with `Matrix`.
# 6. Use `eltype` instead of `where`.
"""
    svd(T::ArrayStorageTensor{<:Number,2}; kwargs...)

svd of an order-2 DenseTensor
"""
function svd(T::ArrayStorageTensor{ElT,2,IndsT}; kwargs...) where {ElT,IndsT}
  truncate = haskey(kwargs, :maxdim) || haskey(kwargs, :cutoff)

  #
  # Keyword argument deprecations
  #
  use_absolute_cutoff = false
  if haskey(kwargs, :absoluteCutoff)
    @warn "In svd, keyword argument absoluteCutoff is deprecated in favor of use_absolute_cutoff"
    use_absolute_cutoff = get(kwargs, :absoluteCutoff, use_absolute_cutoff)
  end

  use_relative_cutoff = true
  if haskey(kwargs, :doRelCutoff)
    @warn "In svd, keyword argument doRelCutoff is deprecated in favor of use_relative_cutoff"
    use_relative_cutoff = get(kwargs, :doRelCutoff, use_relative_cutoff)
  end

  if haskey(kwargs, :fastsvd) || haskey(kwargs, :fastSVD)
    error(
      "In svd, fastsvd/fastSVD keyword arguments are removed in favor of alg, see documentation for more details.",
    )
  end

  maxdim::Int = get(kwargs, :maxdim, minimum(dims(T)))
  mindim::Int = get(kwargs, :mindim, 1)
  cutoff = get(kwargs, :cutoff, 0.0)
  use_absolute_cutoff::Bool = get(kwargs, :use_absolute_cutoff, use_absolute_cutoff)
  use_relative_cutoff::Bool = get(kwargs, :use_relative_cutoff, use_relative_cutoff)
  alg::String = get(kwargs, :alg, "divide_and_conquer")

  #@timeit_debug timer "dense svd" begin
  if alg == "divide_and_conquer"
    MUSV = svd_catch_error(matrix(T); alg=LinearAlgebra.DivideAndConquer())
    if isnothing(MUSV)
      # If "divide_and_conquer" fails, try "qr_iteration"
      alg = "qr_iteration"
      MUSV = svd_catch_error(matrix(T); alg=LinearAlgebra.QRIteration())
      if isnothing(MUSV)
        # If "qr_iteration" fails, try "recursive"
        alg = "recursive"
        MUSV = svd_recursive(matrix(T))
      end
    end
  elseif alg == "qr_iteration"
    MUSV = svd_catch_error(matrix(T); alg=LinearAlgebra.QRIteration())
    if isnothing(MUSV)
      # If "qr_iteration" fails, try "recursive"
      alg = "recursive"
      MUSV = svd_recursive(matrix(T))
    end
  elseif alg == "recursive"
    MUSV = svd_recursive(matrix(T))
  elseif alg == "QRAlgorithm" || alg == "JacobiAlgorithm"
    MUSV = svd_catch_error(matrix(T); alg=alg)
  else
    error(
      "svd algorithm $alg is not currently supported. Please see the documentation for currently supported algorithms.",
    )
  end
  if isnothing(MUSV)
    if any(isnan, T)
      println("SVD failed, the matrix you were trying to SVD contains NaNs.")
    else
      println(lapack_svd_error_message(alg))
    end
    return nothing
  end
  MU, MS, MV = MUSV
  conj!(MV)
  #end # @timeit_debug

  P = MS .^ 2
  if truncate
    P, truncerr, _ = truncate!!(
      P; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff, kwargs...
    )
  else
    truncerr = 0.0
  end
  spec = Spectrum(P, truncerr)
  dS = length(P)
  if dS < length(MS)
    MU = MU[:, 1:dS]
    # Fails on some GPU backends like Metal.
    # resize!(MS, dS)
    MS = MS[1:dS]
    MV = MV[:, 1:dS]
  end

  # Make the new indices to go onto U and V
  u = eltype(IndsT)(dS)
  v = eltype(IndsT)(dS)
  Uinds = IndsT((ind(T, 1), u))
  Sinds = IndsT((u, v))
  Vinds = IndsT((ind(T, 2), v))
  U = tensor(MU, Uinds)
  S = tensor(Diag(MS), Sinds)
  V = tensor(MV, Vinds)
  return U, S, V, spec
end

## function svd(
##   tens::ArrayStorageTensor;
##   alg,
##   which_decomp,
##   tags,
##   mindim,
##   cutoff,
##   eigen_perturbation,
##   normalize,
##   maxdim,
## )
##   error("Not implemented")
##   F = svd(storage(tens))
##   U, S, V = F.U, F.S, F.Vt
##   i, j = inds(tens)
##   # TODO: Make this more general with a `similar_ind` function,
##   # so the dimension can be determined from the length of `S`.
##   min_ij = dim(i) ≤ dim(j) ? i : j
##   α = sim(min_ij) # similar_ind(i, space(S))
##   β = sim(min_ij) # similar_ind(i, space(S))
##   Utensor = tensor(U, (i, α))
##   # TODO: Remove conversion to `Diagonal` to make more general, or make a generic `Diagonal` concept that works for `BlockSparseArray`.
##   # Used for now to avoid introducing wrapper types.
##   Stensor = tensor(Diagonal(S), (α, β))
##   Vtensor = tensor(V, (β, j))
##   return Utensor, Stensor, Vtensor, Spectrum(nothing, 0.0)
## end
