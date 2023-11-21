# https://github.com/invenia/NamedDims.jl
# https://github.com/mcabbott/NamedPlus.jl

abstract type AbstractNamedDimsArray{T,N,Names} <: AbstractArray{T,N} end

# Required interface

# Output the names.
dimnames(a::AbstractNamedDimsArray) = error("Not implemented")

# Unwrapping the names
Base.parent(::AbstractNamedDimsArray) = error("Not implemented")

# Set the names of an unnamed AbstractArray
# `ndims(a) == length(names)`
# This is a constructor
## named(a::AbstractArray, names) = error("Not implemented")

# Traits
isnamed(::AbstractNamedDimsArray) = true

# AbstractArray interface
# TODO: Use `unname` instead of `parent`?

# Helper function, move to `utils.jl`.
named_tuple(t::Tuple, names) = ntuple(i -> named(t[i], names[i]), length(t))

# TODO: Use the proper type, `namedaxistype(a)`.
Base.axes(a::AbstractNamedDimsArray) = named_tuple(axes(unname(a)), dimnames(a))
# TODO: Use the proper type, `namedlengthtype(a)`.
Base.size(a::AbstractNamedDimsArray) = length.(axes(a))
Base.getindex(a::AbstractNamedDimsArray, I...) = unname(a)[I...]
function Base.setindex!(a::AbstractNamedDimsArray, x, I...)
  unname(a)[I...] = x
  return a
end

# Derived interface

# Output the names.
dimname(a::AbstractNamedDimsArray, i) = dimnames(a)[i]

# Renaming
# Unname and set new naems
rename(a::AbstractNamedDimsArray, names) = named(unname(a), names)

# replacenames(a, :i => :a, :j => :b)
# `rename` in `NamedPlus.jl`.
replacenames(a::AbstractNamedDimsArray, names::Pair) = error("Not implemented yet")

# Either define new names or replace names
setnames(a::AbstractArray, names) = named(a, names)
setnames(a::AbstractNamedDimsArray, names) = rename(a, names)

function getperm(x, y)
  return map(xᵢ -> findfirst(isequal(xᵢ), y), x)
end

function get_name_perm(a::AbstractNamedDimsArray, names::Tuple)
  return getperm(dimnames(a), names)
end

# Ambiguity error
function get_name_perm(a::AbstractNamedDimsArray, names::Tuple{})
  @assert iszero(ndims(a))
  return ()
end

function get_name_perm(
  a::AbstractNamedDimsArray, namedints::Tuple{Vararg{AbstractNamedInt}}
)
  return getperm(size(a), namedints)
end

function get_name_perm(
  a::AbstractNamedDimsArray, namedaxes::Tuple{Vararg{AbstractNamedUnitRange}}
)
  return getperm(axes(a), namedaxes)
end

# Indexing
# a[:i => 2, :j => 3]
# TODO: Write a generic version using `dim`.
# TODO: Define a `NamedIndex` type for indexing?
function Base.getindex(a::AbstractNamedDimsArray, I::Pair...)
  perm = get_name_perm(a, first.(I))
  i = last.(I)
  return unname(a)[map(p -> i[p], perm)...]
end

# a[:i => 2, :j => 3] = 12
# TODO: Write a generic version using `dim`.
function Base.setindex!(a::AbstractNamedDimsArray, value, I::Pair...)
  perm = get_name_perm(a, first.(I))
  i = last.(I)
  unname(a)[map(p -> i[p], perm)...] = value
  return a
end

# Output the dimension of the specified name.
dim(a::AbstractNamedDimsArray, name) = findfirst(==(name), dimnames(a))

# Output the dimensions of the specified names.
dims(a::AbstractNamedDimsArray, names) = map(name -> dim(a, name), names)

# Unwrapping the names
unname(a::AbstractNamedDimsArray) = parent(a)
unname(a::AbstractArray) = a

# Permute into a certain order.
# align(a, (:j, :k, :i))
# Like `named(nameless(a, names), names)`
function align(a::AbstractNamedDimsArray, names)
  perm = get_name_perm(a, names)
  # TODO: Avoid permutation if it is a trivial permutation?
  return typeof(a)(permutedims(unname(a), perm), names)
end

# Unwrapping names and permuting
# nameless(a, (:j, :i))
# Could just call `unname`?
## nameless(a::AbstractNamedDimsArray, names) = unname(align(a, names))
unname(a::AbstractNamedDimsArray, names) = unname(align(a, names))

# In `TensorAlgebra` this this `fuse` and `unfuse`,
# in `NDTensors`/`ITensors` this is `combine` and `uncombine`.
# t = split(g, :n => (j=4, k=5))
# join(t, (:i, :k) => :χ)

# TensorAlgebra
# contract, fusedims, unfusedims, qr, eigen, svd, add, etc.
# Some of these can simply wrap `TensorAlgebra.jl` functions.
