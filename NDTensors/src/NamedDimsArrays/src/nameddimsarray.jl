struct NamedDimsArray{T,N,Arr<:AbstractArray{T,N},Names} <:
       AbstractNamedDimsArray{T,N,Names}
  array::Arr
  names::Names
end

# TODO: Check size consistency
function NamedDimsArray{T,N,Arr,Names}(
  a::AbstractArray, namedsize::Tuple{Vararg{AbstractNamedInt}}
) where {T,N,Arr<:AbstractArray{T,N},Names}
  @assert size(a) == unname.(namedsize)
  return NamedDimsArray{T,N,Arr,Names}(a, name.(namedsize))
end

# TODO: Check axes consistency
function NamedDimsArray{T,N,Arr,Names}(
  a::AbstractArray, namedaxes::Tuple{Vararg{AbstractNamedUnitRange}}
) where {T,N,Arr<:AbstractArray{T,N},Names}
  @assert axes(a) == unname.(namedaxes)
  return NamedDimsArray{T,N,Arr,Names}(a, name.(namedaxes))
end

# Required interface

# Output the names.
dimnames(a::NamedDimsArray) = a.names

# Unwrapping the names
Base.parent(a::NamedDimsArray) = a.array

# Set the names of an unnamed AbstractArray
function named(a::AbstractArray, names)
  @assert ndims(a) == length(names)
  return NamedDimsArray(a, names)
end

# TODO: Use `Undefs.jl` instead.
function undefs(arraytype::Type{<:AbstractArray}, axes::Tuple{Vararg{AbstractUnitRange}})
  return arraytype(undef, length.(axes))
end

# TODO: Use `AbstractNamedUnitRange`, determine the `AbstractNamedDimsArray`
# from a default value. Useful for distinguishing between `NamedDimsArray`
# and `ITensor`.
function undefs(arraytype::Type{<:AbstractArray}, axes::Tuple{Vararg{NamedUnitRange}})
  array = undefs(arraytype, unname.(axes))
  names = name.(axes)
  return named(array, names)
end

# TODO: Use `AbstractNamedUnitRange`, determine the `AbstractNamedDimsArray`
# from a default value. Useful for distinguishing between `NamedDimsArray`
# and `ITensor`.
function Base.similar(
  arraytype::Type{<:AbstractArray}, axes::Tuple{NamedUnitRange,Vararg{NamedUnitRange}}
)
  # TODO: Use `unname`?
  return undefs(arraytype, axes)
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
