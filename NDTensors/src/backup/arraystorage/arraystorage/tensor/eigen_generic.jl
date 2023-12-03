function truncate!!(
  d::AbstractVector,
  u::AbstractMatrix;
  mindim,
  maxdim,
  cutoff,
  use_absolute_cutoff,
  use_relative_cutoff,
)
  error("Not implemented")
  # Sort by largest to smallest eigenvalues
  # TODO: Replace `cpu` with `Expose` dispatch.
  p = sortperm(cpu(d); rev=true, by=abs)
  d = d[p]
  u = u[:, p]

  if any(!isnothing, (maxdim, cutoff))
    d, truncerr, _ = truncate!!(
      d; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff
    )
    length_d = length(d)
    if length_d < size(u, 2)
      u = u[:, 1:length_d]
    end
  else
    length_d = length(d)
    # TODO: Make this `zero(eltype(d))`?
    truncerr = 0.0
  end
  spec = Spectrum(d, truncerr)
  return d, u, spec
end

# TODO: Rewrite this function to be more modern:
# 1. List keyword arguments in function signature.
# 2. Output `Spectrum` as a keyword argument that gets overwritten.
# 3. Make this into two layers, one that handles indices and one that works with `AbstractMatrix`.
function LinearAlgebra.eigen(
  t::MatrixOrArrayStorageTensor;
  maxdim=nothing,
  mindim=nothing,
  cutoff=nothing,
  use_absolute_cutoff=nothing,
  use_relative_cutoff=nothing,
  # These are getting passed erroneously.
  # TODO: Make sure they don't get passed down
  # to here.
  which_decomp=nothing,
  tags=nothing,
  eigen_perturbation=nothing,
  normalize=nothing,
  ishermitian=nothing,
  ortho=nothing,
  svd_alg=nothing,
)
  a = storage(t)
  ## TODO Here I am calling parent to ensure that the correct `any` function
  ## is envoked for non-cpu matrices
  if any(!isfinite, parent(a))
    throw(
      ArgumentError(
        "Trying to perform the eigendecomposition of a matrix containing NaNs or Infs"
      ),
    )
  end
  d, u = eigen(a)
  d, u, spec = truncate!!(d, u)

  error("Not implemented")

  # Make the new indices to go onto V
  # TODO: Put in a separate function, such as
  # `rewrap_inds` or something like that.
  indstype = typeof(inds(t))

  # TODO: Make this generic to dense or block sparse.
  l = eltype(indstype)(axes(t, 1))
  r = eltype(indstype)(axes(t, 2))
  inds_d = indstype((l, dag(r)))
  inds_u = indstype((dag(ind(T, 2)), dag(r)))
  dₜ = tensor(DiagonalMatrix(DM), Dinds)
  uₜ = tensor(u, u_inds)
  return dₜ, uₜ, spec
end
