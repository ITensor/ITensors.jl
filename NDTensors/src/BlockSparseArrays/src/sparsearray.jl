# TODO: Define a constructor with a default `zero`.
struct SparseArray{T,N,Zero} <: AbstractArray{T,N}
  data::Dictionary{CartesianIndex{N},T}
  dims::NTuple{N,Int}
  zero::Zero
end

default_zero() = (eltype, I) -> zero(eltype)

SparseArray{T}(size::Tuple{Vararg{Integer}}, zero=default_zero()) where {T} = SparseArray(Dictionary{CartesianIndex{length(size)},T}(), size, zero)
SparseArray{T}(size::Integer...) where {T} = SparseArray{T}(size)

SparseArray{T}(axes::Tuple{Vararg{AbstractUnitRange}}, zero=default_zero()) where {T} = SparseArray{T}(length.(axes), zero)
SparseArray{T}(axes::AbstractUnitRange...) where {T} = SparseArray{T}(length.(axes))

Base.size(a::SparseArray) = a.dims

function Base.setindex!(a::SparseArray{T,N}, v, I::CartesianIndex{N}) where {T,N}
  set!(a.data, I, v)
  return a
end
function Base.setindex!(a::SparseArray{T,N}, v, I::Vararg{Int,N}) where {T,N}
  return setindex!(a, v, CartesianIndex(I))
end

function Base.getindex(a::SparseArray{T,N}, I::CartesianIndex{N}) where {T,N}
  # Don't evaluate `a.zero` unless it is needed.
  if !isassigned(a.data, I)
    return a.zero(T, I)
  end
  return a.data[I]
end
function Base.getindex(a::SparseArray{T,N}, I::Vararg{Int,N}) where {T,N}
  return getindex(a, CartesianIndex(I))
end

# SparseArrayKit.jl syntax
nonzero_keys(a::SparseArray) = keys(a.data)

function Base.reshape(
  a::SparseArray{T,N,Zero}, dims::Tuple{Vararg{Int,M}}
) where {T,N,M,Zero}
  a_reshaped = SparseArray{T,M,Zero}(Dictionary{CartesianIndex{M},T}(), dims, a.zero)
  copyto!(a_reshaped, Base.ReshapedArray(a, dims, ()))
  for I in nonzero_keys(a)
    i = LinearIndices(a)[I]
    I_reshaped = CartesianIndices(a_reshaped)[i]
    a_reshaped[I_reshaped] = a[I]
  end
  return a_reshaped
end

# TODO: Make `PermutedSparseArray`.
# TODO: Distinguish between preserving structure or not.
# TODO: Make `similar`, either empty or preserving structure.
function Base.zero(a::PermutedDimsArray{<:Any,<:Any,<:Any,<:Any,<:SparseArray})
  # TODO: Make a simpler empty constructor.
  # SparseArray(size(a), parent(a).zero)
  # TODO: Define `zero_elt`.
  return SparseArray(
    Dictionary{CartesianIndex{ndims(a)},eltype(a)}(), size(a), parent(a).zero
  )
end

# TODO: Distinguish between preserving structure or not.
# TODO: Make `similar`, either empty or preserving structure.
# TODO: Combine with definition above.
function Base.zero(a::SparseArray)
  # TODO: Make a simpler empty constructor.
  # SparseArray(size(a), parent(a).zero)
  # TODO: Define `zero_elt`.
  return SparseArray(Dictionary{CartesianIndex{ndims(a)},eltype(a)}(), size(a), a.zero)
end

const SparseArrayLike{T,N,Zero} = Union{
  SparseArray{T,N,Zero},PermutedDimsArray{T,N,<:Any,<:Any,SparseArray{T,N,Zero}}
}

# TODO: Include `PermutedDimsArray`.
# TODO: Define `SparseArrayLike`.
function map_nonzeros!(f, a_dest::AbstractArray, as::SparseArrayLike...)
  @assert allequal(axes.(as))
  # TODO: Define `nonzero_keys` for multiple arrays, maybe with union?
  for index in union(nonzero_keys.(as)...)
    a_dest[index] = f(map(a -> a[index], as)...)
  end
  return a_dest
end

## function output_type(f, args::Type...)
##   # TODO: Is this good to use here?
##   # Seems best for `Number` subtypes, maybe restrict to that here.
##   return Base.promote_op(f, args...)
## end
## 
## function output_eltype(::typeof(map_nonzeros), fmap, as::Type{<:AbstractArray}...)
##   return output_type(fmap, eltype.(as)...)
## end
## 
## function output_eltype(f::typeof(map_nonzeros), fmap, as::AbstractArray...)
##   return output_eltype(f, fmap, typeof.(as)...)
## end
## 
## function output_structure(f::typeof(map_nonzeros), fmap, as::SparseArray...)
## end
## 
## function output_structure(f::typeof(map_nonzeros), fmap, as::AbstractArray...)
##   return ArrayStructure(; eltype=output_eltype(f, fmap, as...), axes=output_axes(f, fmap, as...))
## end
## 
## function output_type(f::typeof(map_nonzeros), fmap, as::AbstractArray...)
##   return error("Not implemented")
## end
## 
## function output_type(f::typeof(map_nonzeros), fmap, as::SparseArrayLike...)
##   return SparseArray
## end
## 
## # Allocate an array with uninitialized/undefined memory
## # according the array type and structure (for example the
## # size or axes).
## function allocate(arraytype::Type{<:AbstractArray}, structure)
##   return arraytype(undef, structure)
## end
## 
## function allocate_zeros(arraytype::Type{<:AbstractArray}, structure)
##   a = allocate(arraytype, structure)
##   # TODO: Use `zeros!!` or `zerovector!!` from VectorInterface.jl.
##   zeros!(a)
##   return a
## end
## 
## function allocate_output(f::typeof(map_nonzeros), fmap, as::AbstractArray...)
##   return allocate_zeros(output_type(f, fmap, as...), output_structure(f, fmap, as...))
## end
## 
function map_nonzeros(f, as::SparseArrayLike...)
  ## @assert allequal(axes.(as))
  # Preserves the element type:
  # a_dest = zero(first(as))
  a_dest = allocate_output(map_nonzeros, f, as...)
  map!(f, a_dest, as...)
  return a_dest
end

function Base.map!(f, a_dest::AbstractArray, as::SparseArrayLike...)
  # TODO: Check `f` preserves zero.
  map_nonzeros!(f, a_dest, as...)
  return a_dest
end

function Base.map(f, as::SparseArrayLike...)
  # TODO: Check `f` preserves zero.
  return map_nonzeros(f, as...)
end
