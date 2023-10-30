# TODO: Rewrite this function to be more modern:
# 1. Output `Spectrum` as a keyword argument that gets overwritten.
# 2. Dispatch on `alg`.
# 3. Make this into two layers, one that handles indices and one that works with `Matrix`.
"""
    svd(T::ArrayStorageTensor{<:Number,2}; kwargs...)

svd of an order-2 DenseTensor
"""
function svd(
  T::ArrayStorageTensor;
  maxdim=nothing,
  mindim=1,
  cutoff=nothing,
  alg="divide_and_conquer", # TODO: Define `default_alg(T)`
  use_absolute_cutoff=false,
  use_relative_cutoff=true,
  # These are getting passed erroneously.
  # TODO: Make sure they don't get passed down
  # to here.
  which_decomp=nothing,
  tags=nothing,
  eigen_perturbation=nothing,
  normalize=nothing,
)
  truncate = !isnothing(maxdim) || !isnothing(cutoff)
  # TODO: Define `default_maxdim(T)`.
  maxdim = isnothing(maxdim) ? minimum(dims(T)) : maxdim
  # TODO: Define `default_cutoff(T)`.
  cutoff = isnothing(cutoff) ? zero(eltype(T)) : cutoff

  # TODO: Dispatch on `Algorithm(alg)`.
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

  P = MS .^ 2
  if truncate
    P, truncerr, _ = truncate!!(
      P; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff
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
  # TODO: Put in a separate function, such as
  # `rewrap_inds` or something like that.
  indstype = typeof(inds(T))
  u = eltype(indstype)(dS)
  v = eltype(indstype)(dS)
  Uinds = indstype((ind(T, 1), u))
  Sinds = indstype((u, v))
  Vinds = indstype((ind(T, 2), v))
  U = tensor(MU, Uinds)
  S = tensor(DiagonalMatrix(MS), Sinds)
  V = tensor(MV, Vinds)
  return U, S, V, spec
end
