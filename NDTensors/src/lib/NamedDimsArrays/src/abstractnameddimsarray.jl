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

# TODO: Should `axes` output named axes or not?
# TODO: Use the proper type, `namedaxistype(a)`.
# Base.axes(a::AbstractNamedDimsArray) = named_tuple(axes(unname(a)), dimnames(a))
Base.axes(a::AbstractNamedDimsArray) = axes(unname(a))
namedaxes(a::AbstractNamedDimsArray) = named.(axes(unname(a)), dimnames(a))
# TODO: Use the proper type, `namedlengthtype(a)`.
Base.size(a::AbstractNamedDimsArray) = size(unname(a))
namedsize(a::AbstractNamedDimsArray) = named.(size(unname(a)), dimnames(a))
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
function replacenames(na::AbstractNamedDimsArray, replacements::Pair...)
  # `BaseExtension.replace` needed for `Tuple` support on Julia 1.6 and older.
  return named(unname(na), BaseExtensions.replace(dimnames(na), replacements...))
end

# Either define new names or replace names
setnames(a::AbstractArray, names) = named(a, names)
setnames(a::AbstractNamedDimsArray, names) = rename(a, names)

# TODO: Move to `utils.jl` file.
function getperm(x, y)
  return map(xᵢ -> findfirst(isequal(xᵢ), y), x)
end

# TODO: Use `isnamed` trait?
function get_name_perm(a::AbstractNamedDimsArray, names::Tuple)
  # TODO: Call `getperm(dimnames(a), dimnames(namedints))`.
  return getperm(dimnames(a), names)
end

# Fixes ambiguity error
# TODO: Use `isnamed` trait?
function get_name_perm(a::AbstractNamedDimsArray, names::Tuple{})
  # TODO: Call `getperm(dimnames(a), dimnames(namedints))`.
  @assert iszero(ndims(a))
  return ()
end

# TODO: Use `isnamed` trait?
function get_name_perm(
  a::AbstractNamedDimsArray, namedints::Tuple{Vararg{AbstractNamedInt}}
)
  # TODO: Call `getperm(dimnames(a), dimnames(namedints))`.
  return getperm(namedsize(a), namedints)
end

# TODO: Use `isnamed` trait?
function get_name_perm(
  a::AbstractNamedDimsArray, new_namedaxes::Tuple{Vararg{AbstractNamedUnitRange}}
)
  # TODO: Call `getperm(dimnames(a), dimnames(namedints))`.
  return getperm(namedaxes(a), new_namedaxes)
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
function align(na::AbstractNamedDimsArray, names)
  perm = get_name_perm(na, names)
  # TODO: Avoid permutation if it is a trivial permutation?
  # return typeof(a)(permutedims(unname(a), perm), names)
  return permutedims(na, perm)
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
