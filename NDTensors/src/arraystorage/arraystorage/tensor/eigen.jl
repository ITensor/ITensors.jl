# TODO: Rewrite this function to be more modern:
# 1. List keyword arguments in function signature.
# 2. Output `Spectrum` as a keyword argument that gets overwritten.
# 3. Dispatch on `alg`.
# 4. Remove keyword argument deprecations.
# 5. Make this into two layers, one that handles indices and one that works with `Matrix`.
# 6. Use `eltype` instead of `where`.
function eigen(
  T::Hermitian{ElT,<:ArrayStorageTensor{ElT,2,IndsT}}; kwargs...
) where {ElT<:Union{Real,Complex},IndsT}
  # Keyword argument deprecations
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

  truncate = haskey(kwargs, :maxdim) || haskey(kwargs, :cutoff)
  maxdim::Int = get(kwargs, :maxdim, minimum(dims(T)))
  mindim::Int = get(kwargs, :mindim, 1)
  cutoff::Union{Nothing,Float64} = get(kwargs, :cutoff, 0.0)
  use_absolute_cutoff::Bool = get(kwargs, :use_absolute_cutoff, use_absolute_cutoff)
  use_relative_cutoff::Bool = get(kwargs, :use_relative_cutoff, use_relative_cutoff)

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
      DM; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff, kwargs...
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
  l = eltype(IndsT)(dD)
  r = eltype(IndsT)(dD)
  Vinds = IndsT((dag(ind(T, 2)), dag(r)))
  Dinds = IndsT((l, dag(r)))
  V = tensor(VM, Vinds)
  D = tensor(Diag(DM), Dinds)
  return D, V, spec
end
