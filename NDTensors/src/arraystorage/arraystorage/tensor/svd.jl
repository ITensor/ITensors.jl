backup_svd_alg(::Algorithm"divide_and_conquer") = Algorithm"qr_iteration"()
backup_svd_alg(::Algorithm"qr_iteration") = Algorithm"recursive"()

function svd(alg::Algorithm"divide_and_conquer", a::ArrayStorage)
  USV = svd_catch_error(a; alg=LinearAlgebra.DivideAndConquer())
  if isnothing(USV)
    return svd(backup_svd_alg(alg), a)
  end
  return USV
end

function svd(alg::Algorithm"qr_iteration", a::ArrayStorage)
  USV = svd_catch_error(a; alg=LinearAlgebra.QRIteration())
  if isnothing(USV)
    return svd(backup_svd_alg(alg), a)
  end
  return USV
end

function svd(alg::Algorithm"recursive", a::ArrayStorage)
  return svd_recursive(a)
end

function svd(::Algorithm"QRAlgorithm", a::ArrayStorage)
  return error("Not implemented yet")
end

function svd(::Algorithm"JacobiAlgorithm", a::ArrayStorage)
  return error("Not implemented yet")
end

function svd(alg::Algorithm, a::ArrayStorage)
  return error(
    "svd algorithm $alg is not currently supported. Please see the documentation for currently supported algorithms.",
  )
end

"""
    tsvd(a::ArrayStorage{<:Number,2}; kwargs...)

svd of an order-2 DenseTensor
"""
function tsvd(
  a::ArrayStorage;
  mindim=nothing,
  maxdim=nothing,
  cutoff=nothing,
  alg=nothing,
  use_absolute_cutoff=nothing,
  use_relative_cutoff=nothing,
  # Only used by BlockSparse svd
  min_blockdim=nothing,
)
  alg = replace_nothing(alg, default_svd_alg(a, unwrap_type(a)))
  USV = svd(Algorithm(alg), a)
  if isnothing(USV)
    if any(isnan, a)
      println("SVD failed, the matrix you were trying to SVD contains NaNs.")
    else
      println(lapack_svd_error_message(alg))
    end
    return nothing
  end

  U, S, V = USV
  conj!(V)

  P = S .^ 2
  if any(!isnothing, (maxdim, cutoff))
    P, truncerr, _ = truncate!!(
      P; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff
    )
  else
    truncerr = 0.0
  end
  spec = Spectrum(P, truncerr)
  dS = length(P)
  if dS < length(S)
    U = U[:, 1:dS]
    # Fails on some GPU backends like Metal.
    # resize!(MS, dS)
    S = S[1:dS]
    V = V[:, 1:dS]
  end
  return U, DiagonalMatrix(S), V, spec
end

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
  mindim=nothing,
  maxdim=nothing,
  cutoff=nothing,
  alg=nothing,
  use_absolute_cutoff=nothing,
  use_relative_cutoff=nothing,
  # Only used by BlockSparse svd
  min_blockdim=nothing,
)
  U, S, V, spec = tsvd(
    storage(T); mindim, maxdim, cutoff, alg, use_absolute_cutoff, use_relative_cutoff
  )
  # Make the new indices to go onto U and V
  # TODO: Put in a separate function, such as
  # `rewrap_inds` or something like that.
  dS = length(S[DiagIndices()])
  indstype = typeof(inds(T))
  u = eltype(indstype)(dS)
  v = eltype(indstype)(dS)
  Uinds = indstype((ind(T, 1), u))
  Sinds = indstype((u, v))
  Vinds = indstype((ind(T, 2), v))
  TU = tensor(U, Uinds)
  TS = tensor(S, Sinds)
  TV = tensor(V, Vinds)
  return TU, TS, TV, spec
end
