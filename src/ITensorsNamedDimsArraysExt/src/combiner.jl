# Combiner
using ..NDTensors.NamedDimsArrays: AbstractNamedDimsArray, dimnames, name, unname
using ..NDTensors.TensorAlgebra: TensorAlgebra, fusedims, splitdims
using NDTensors: NDTensors, Tensor, Combiner

# TODO: Move to `NamedDimsArraysTensorAlgebraExt`.
using ..ITensors: IndexID
using LinearAlgebra: LinearAlgebra, qr
using ..NDTensors.NamedDimsArrays: AbstractNamedDimsArray, dimnames, name, unname
function LinearAlgebra.qr(na::AbstractNamedDimsArray; positive=nothing)
  # TODO: Make this more systematic.
  i, j = dimnames(na)
  # TODO: Create a `TensorAlgebra.qr`.
  q, r = qr(unname(na))
  # TODO: Use `sim` or `rand(::IndexID)`.
  name_qr = IndexID(rand(UInt64), "", 0)
  # TODO: Make this GPU-friendly.
  nq = named(Matrix(q), (i, name_qr))
  nr = named(Matrix(r), (name_qr, j))
  return nq, nr
end

# TODO: Move to `NamedDimsArraysTensorAlgebraExt`.
using ..ITensors: IndexID
using LinearAlgebra: LinearAlgebra, Hermitian, eigen
using ..NDTensors.DiagonalArrays: DiagonalMatrix
using ..NDTensors.NamedDimsArrays: AbstractNamedDimsArray, dimnames, name, unname
using ..NDTensors: Spectrum, truncate!!
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
  l = IndexID(rand(UInt64), "", 0)
  r = IndexID(rand(UInt64), "", 0)
  names_d = (l, r)
  nd = named(DiagonalMatrix(d), names_d)
  names_u = (names_a[2], r)
  nu = named(u, names_u)
  return nd, nu, spec
end

# TODO: Move to `NamedDimsArraysTensorAlgebraExt`.
function TensorAlgebra.fusedims(na::AbstractNamedDimsArray, fusions::Pair...)
  # TODO: generalize to multiple fused groups of dimensions
  @assert isone(length(fusions))
  fusion = only(fusions)

  split_names = first(fusion)
  fused_name = last(fusion)

  split_dims = map(split_name -> findfirst(isequal(split_name), dimnames(na)), split_names)
  fused_dim = findfirst(isequal(fused_name), dimnames(na))
  @assert isnothing(fused_dim)

  unfused_dims = Tuple.(setdiff(1:ndims(na), split_dims))
  partitioned_perm = (unfused_dims..., split_dims)

  a_fused = fusedims(unname(na), partitioned_perm...)
  names_fused = (setdiff(dimnames(na), split_names)..., fused_name)
  return named(a_fused, names_fused)
end

function TensorAlgebra.splitdims(na::AbstractNamedDimsArray, splitters::Pair...)
  fused_names = first.(splitters)
  split_namedlengths = last.(splitters)
  splitters_unnamed = map(splitters) do splitter
    fused_name, split_namedlengths = splitter
    fused_dim = findfirst(isequal(fused_name), dimnames(na))
    split_lengths = unname.(split_namedlengths)
    return fused_dim => split_lengths
  end
  a_split = splitdims(unname(na), splitters_unnamed...)
  names_split = Any[tuple.(dimnames(na))...]
  for splitter in splitters
    fused_name, split_namedlengths = splitter
    fused_dim = findfirst(isequal(fused_name), dimnames(na))
    split_names = name.(split_namedlengths)
    names_split[fused_dim] = split_names
  end
  names_split = reduce((x, y) -> (x..., y...), names_split)
  return named(a_split, names_split)
end

function ITensors._contract(na::AbstractNamedDimsArray, c::Tensor{<:Any,<:Any,<:Combiner})
  split_names = name.(NDTensors.uncombinedinds(c))
  fused_name = name(NDTensors.combinedind(c))

  # Use to determine if we are doing fusion or splitting.
  split_dims = map(split_name -> findfirst(isequal(split_name), dimnames(na)), split_names)
  fused_dim = findfirst(isequal(fused_name), dimnames(na))

  return if isnothing(fused_dim)
    # Dimension fusion (joining, combining)
    @assert all(!isnothing, split_dims)
    fusedims(na, split_names => fused_name)
  else
    # Dimension unfusion (splitting, uncombining)
    @assert all(isnothing, split_dims)

    split_dims = NamedInt.(NDTensors.uncombinedinds(c))
    splitdims(na, fused_name => split_dims)
  end
end
