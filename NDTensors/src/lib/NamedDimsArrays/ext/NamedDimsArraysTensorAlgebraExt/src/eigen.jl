## using ..ITensors: IndexID
using LinearAlgebra: LinearAlgebra, Diagonal, Hermitian, eigen
## using ..NDTensors.DiagonalArrays: DiagonalMatrix
using ...NDTensors.NamedDimsArrays: AbstractNamedDimsArray, dimnames, name, unname
using ...NDTensors.RankFactorization: Spectrum, truncate!!
function LinearAlgebra.eigen(
  na::Hermitian{T,<:AbstractNamedDimsArray{T}};
  mindim=nothing,
  maxdim=nothing,
  cutoff=nothing,
  use_absolute_cutoff=nothing,
  use_relative_cutoff=nothing,
) where {T<:Union{Real,Complex}}
  # TODO: Handle array wrappers around
  # `AbstractNamedDimsArray` more elegantly.
  d, u = eigen(Hermitian(unname(parent(na))))

  # Sort by largest to smallest eigenvalues
  # TODO: Replace `cpu` with `Expose` dispatch.
  p = sortperm(d; rev=true, by=abs)
  d = d[p]
  u = u[:, p]

  length_d = length(d)
  truncerr = zero(Float64) # Make more generic
  if any(!isnothing, (maxdim, cutoff))
    d, truncerr, _ = truncate!!(
      d; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff
    )
    length_d = length(d)
    if length_d < size(u, 2)
      u = u[:, 1:length_d]
    end
  end
  spec = Spectrum(d, truncerr)

  # TODO: Handle array wrappers more generally.
  names_a = dimnames(parent(na))
  # TODO: Make this more generic, handle `dag`, etc.
  l = randname(names_a[1]) # IndexID(rand(UInt64), "", 0)
  r = randname(names_a[2]) # IndexID(rand(UInt64), "", 0)
  names_d = (l, r)
  nd = named(Diagonal(d), names_d)
  names_u = (names_a[2], r)
  nu = named(u, names_u)
  return nd, nu, spec
end
