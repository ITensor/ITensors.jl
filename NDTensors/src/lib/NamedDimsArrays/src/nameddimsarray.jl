function _NamedDimsArray end

struct NamedDimsArray{T,N,Arr<:AbstractArray{T,N},Names} <:
       AbstractNamedDimsArray{T,N,Names}
  array::Arr
  names::Names
  global @inline function _NamedDimsArray(array::AbstractArray, names)
    elt = eltype(array)
    n = ndims(array)
    arraytype = typeof(array)
    namestype = typeof(names)
    return new{elt,n,arraytype,namestype}(array, names)
  end

  # TODO: Delete
  global @inline function _NamedDimsArray(array::NamedDimsArray, names)
    return error("Not implemented, already named.")
  end
end

function NamedDimsArray{T,N,Arr,Names}(
  a::AbstractArray, names
) where {T,N,Arr<:AbstractArray{T,N},Names}
  return _NamedDimsArray(convert(Arr, a), convert(Names, names))
end

# TODO: Combine with other constructor definitions.
function NamedDimsArray{T,N,Arr,Names}(
  a::AbstractArray, names::Tuple{}
) where {T,N,Arr<:AbstractArray{T,N},Names}
  return _NamedDimsArray(convert(Arr, a), convert(Names, names))
end

NamedDimsArray(a::AbstractArray, names) = _NamedDimsArray(a, names)

# TODO: Check size consistency
# TODO: Combine with other constructor definitions.
function NamedDimsArray{T,N,Arr,Names}(
  a::AbstractArray, namedsize::Tuple{Vararg{AbstractNamedInt}}
) where {T,N,Arr<:AbstractArray{T,N},Names}
  @assert size(a) == unname.(namedsize)
  return _NamedDimsArray(convert(Arr, a), convert(Names, name.(namedsize)))
end

# TODO: Check axes consistency
# TODO: Combine with other constructor definitions.
function NamedDimsArray{T,N,Arr,Names}(
  a::AbstractArray, namedaxes::Tuple{Vararg{AbstractNamedUnitRange}}
) where {T,N,Arr<:AbstractArray{T,N},Names}
  @assert axes(a) == unname.(namedaxes)
  return _NamedDimsArray(convert(Arr, a), convert(Names, name.(namedaxes)))
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
