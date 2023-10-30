# TODO: Rewrite this function to be more modern:
# 1. List keyword arguments in function signature.
# 2. Output `Spectrum` as a keyword argument that gets overwritten.
# 3. Make this into two layers, one that handles indices and one that works with `AbstractMatrix`.
function eigen(
  T::Hermitian{<:Any,<:ArrayStorageTensor};
  use_absolute_cutoff=false,
  use_relative_cutoff=true,
  maxdim=nothing,
  mindim=1,
  cutoff=nothing,
  # These are getting passed erroneously.
  # TODO: Make sure they don't get passed down
  # to here.
  ishermitian=nothing,
  which_decomp=nothing,
  tags=nothing,
  eigen_perturbation=nothing,
  ortho=nothing,
  normalize=nothing,
  svd_alg=nothing,
)
  truncate = !isnothing(maxdim) || !isnothing(cutoff)
  maxdim = isnothing(maxdim) ? minimum(dims(T)) : maxdim
  cutoff = isnothing(cutoff) ? zero(eltype(T)) : cutoff

  matrixT = matrix(T)
  ## TODO Here I am calling parent to ensure that the correct `any` function
  ## is envoked for non-cpu matrices
  if any(!isfinite, parent(matrixT))
    throw(
      ArgumentError(
        "Trying to perform the eigendecomposition of a matrix containing NaNs or Infs"
      ),
    )
  end

  DM, VM = eigen(matrixT)

  # Sort by largest to smallest eigenvalues
  # TODO: Replace `cpu` with `leaf_parenttype` dispatch.
  p = sortperm(cpu(DM); rev=true, by=abs)
  DM = DM[p]
  VM = VM[:, p]

  if truncate
    DM, truncerr, _ = truncate!!(
      DM; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff
    )
    dD = length(DM)
    if dD < size(VM, 2)
      VM = VM[:, 1:dD]
    end
  else
    dD = length(DM)
    truncerr = 0.0
  end
  spec = Spectrum(DM, truncerr)

  # Make the new indices to go onto V
  # TODO: Put in a separate function, such as
  # `rewrap_inds` or something like that.
  indstype = typeof(inds(T))
  l = eltype(indstype)(dD)
  r = eltype(indstype)(dD)
  Vinds = indstype((dag(ind(T, 2)), dag(r)))
  Dinds = indstype((l, dag(r)))
  V = tensor(VM, Vinds)
  # TODO: Replace with `DiagonalArray`.
  D = tensor(Diag(DM), Dinds)
  return D, V, spec
end
