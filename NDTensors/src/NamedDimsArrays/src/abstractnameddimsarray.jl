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
named(a::AbstractArray, names) = error("Not implemented")

# Derived interface

# Renaming
# Unname and set new naems
rename(a::NamedDimsArray, names) = named(unname(a), names)

# replacenames(a, :i => :a, :j => :b)
# `rename` in `NamedPlus.jl`.
replacenames(a::AbstractNamedDimsArray, names::Pair) = error("Not implemented yet")

# Either define new names or replace names
setnames(a::AbstractArray, names) = named(a, names)
setnames(a::AbstractNamedDimsArray, names) = rename(a, names)

# Indexing
# a[:i => 2, :j => 3]
# TODO: Write a generic version using `dim`.
Base.getindex(a::AbstractNamedDimsArray, I::Pair...) = error("Not implemented yet")

# a[:i => 2, :j => 3] = 12
# TODO: Write a generic version using `dim`.
Base.setindex!(a::AbstractNamedDimsArray, x, I::Pair...) = error("Not implemented yet")

# Output the dimension of the specified name.
dim(a::AbstractNamedDimsArray, name) = findfirst(==(name), dimnames(a))

# Output the dimensions of the specified names.
dims(a::AbstractNamedDimsArray, names) = map(name -> dim(a, name), names)

# Unwrapping the names
unname(a::AbstractNamedDimsArray) = parent(a)
unname(a::AbstractArray) = a

# Unwrapping names and permuting
# nameless(a, (:j, :i))
nameless(a::AbstractNamedDimsArray, names) = error("Not implemented yet")

# In `TensorAlgebra` this this `fuse` and `unfuse`,
# in `NDTensors`/`ITensors` this is `combine` and `uncombine`.
# t = split(g, :n => (j=4, k=5))
# join(t, (:i, :k) => :χ)

# Permute into a certain order.
# align(a, (:j, :k, :i))
# Like `named(nameless(a, names), names)`

# TensorAlgebra
# contract, fusedims, unfusedims, qr, eigen, svd, add, etc.
# Some of these can simply wrap `TensorAlgebra.jl` functions.
