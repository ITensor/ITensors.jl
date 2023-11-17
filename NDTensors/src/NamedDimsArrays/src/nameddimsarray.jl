struct NamedDimsArray{T,N,Arr<:AbstractArray{T,N},Names} <: AbstractNamedDimsArray{T,N,Names}
  array::Arr
  names::Names
end

# Required interface

# Output the names.
dimnames(a::NamedDimsArray) = a.names

# Unwrapping the names
Base.parent(::NamedDimsArray) = a.array

# Set the names of an unnamed AbstractArray
function named(a::AbstractArray, names)
  @assert ndims(a) == length(names)
  return NamedDimsArray(a, names)
end

# TODO: Define `NamedInt`, `NamedUnitRange`, `NamedVector <: AbstractVector`, etc.
# See https://github.com/mcabbott/NamedPlus.jl/blob/v0.0.5/src/int.jl.

# TODO: Define `similar_name`, with shorthand `sim`, that makes a random name.
# Used in matrix/tensor factorizations.

# TODO: Think about how to interact with array wrapper types, see:
# https://github.com/mcabbott/NamedPlus.jl/blob/v0.0.5/src/recursion.jl

# TODO: What should `size` and `axes` output? Could output tuples
# of `NamedInt` and `NamedUnitRange`.

# TODO: Construct from `NamedInt` or `NamedUnitRange` in standard
# array constructors, like `zeros`, `rand`, `randn`, `undefs`, etc.
# See https://mkitti.github.io/Undefs.jl/stable/,
# https://github.com/mkitti/ArrayAllocators.jl

# TODO: Define `ArrayConstructors.randn`, `ArrayConstructors.rand`,
# `ArrayConstructors.zeros`, `ArrayConstructors.fill`, etc.
# for generic constructors accepting `CuArray`, `Array`, etc.
# Also could defign allocator types, `https://github.com/JuliaGPU/KernelAbstractions.jl`
# and `https://docs.juliahub.com/General/HeterogeneousComputing/stable/`.
