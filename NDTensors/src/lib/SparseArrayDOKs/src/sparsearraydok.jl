using Accessors: @set
using Dictionaries: Dictionary, set!
using ..SparseArrayInterface:
  SparseArrayInterface, AbstractSparseArray, getindex_zero_function

# TODO: Parametrize by `data`?
mutable struct SparseArrayDOK{T,N,Zero} <: AbstractSparseArray{T,N}
  data::Dictionary{CartesianIndex{N},T}
  dims::NTuple{N,Int}
  zero::Zero
end

# Constructors
function SparseArrayDOK{T,N,Zero}(dims::Tuple{Vararg{Int}}, zero) where {T,N,Zero}
  return SparseArrayDOK{T,N,Zero}(default_data(T, N), dims, zero)
end

function SparseArrayDOK{T,N}(dims::Tuple{Vararg{Int}}, zero) where {T,N}
  return SparseArrayDOK{T,N,typeof(zero)}(dims, zero)
end

function SparseArrayDOK{T,N}(dims::Tuple{Vararg{Int}}) where {T,N}
  return SparseArrayDOK{T,N}(dims, default_zero())
end

function SparseArrayDOK{T}(dims::Tuple{Vararg{Int}}) where {T}
  return SparseArrayDOK{T,length(dims)}(dims)
end

function SparseArrayDOK{T}(dims::Int...) where {T}
  return SparseArrayDOK{T}(dims)
end

# Specify zero function
function SparseArrayDOK{T}(dims::Tuple{Vararg{Int}}, zero) where {T}
  return SparseArrayDOK{T,length(dims)}(dims, zero)
end

# undef
function SparseArrayDOK{T,N,Zero}(
  ::UndefInitializer, dims::Tuple{Vararg{Int}}, zero
) where {T,N,Zero}
  return SparseArrayDOK{T,N,Zero}(dims, zero)
end

function SparseArrayDOK{T,N}(::UndefInitializer, dims::Tuple{Vararg{Int}}, zero) where {T,N}
  return SparseArrayDOK{T,N}(dims, zero)
end

function SparseArrayDOK{T,N}(::UndefInitializer, dims::Tuple{Vararg{Int}}) where {T,N}
  return SparseArrayDOK{T,N}(dims)
end

function SparseArrayDOK{T}(::UndefInitializer, dims::Tuple{Vararg{Int}}) where {T}
  return SparseArrayDOK{T}(dims)
end

# Axes version
function SparseArrayDOK{T}(
  ::UndefInitializer, axes::Tuple{Vararg{AbstractUnitRange}}
) where {T}
  @assert all(isone, first.(axes))
  return SparseArrayDOK{T}(length.(axes))
end

function SparseArrayDOK{T}(::UndefInitializer, dims::Int...) where {T}
  return SparseArrayDOK{T}(dims...)
end

function SparseArrayDOK{T}(::UndefInitializer, dims::Tuple{Vararg{Int}}, zero) where {T}
  return SparseArrayDOK{T}(dims, zero)
end

# Base `AbstractArray` interface
Base.size(a::SparseArrayDOK) = a.dims

SparseArrayInterface.getindex_zero_function(a::SparseArrayDOK) = a.zero
function SparseArrayInterface.set_getindex_zero_function(a::SparseArrayDOK, f)
  return @set a.zero = f
end

function SparseArrayInterface.setindex_notstored!(
  a::SparseArrayDOK{<:Any,N}, value, I::CartesianIndex{N}
) where {N}
  set!(SparseArrayInterface.sparse_storage(a), I, value)
  return a
end

function Base.similar(a::SparseArrayDOK, elt::Type, dims::Tuple{Vararg{Int}})
  return SparseArrayDOK{elt}(undef, dims, getindex_zero_function(a))
end

# `SparseArrayInterface` interface
SparseArrayInterface.sparse_storage(a::SparseArrayDOK) = a.data

function SparseArrayInterface.dropall!(a::SparseArrayDOK)
  return empty!(SparseArrayInterface.sparse_storage(a))
end

SparseArrayDOK(a::AbstractArray) = SparseArrayDOK{eltype(a)}(a)

SparseArrayDOK{T}(a::AbstractArray) where {T} = SparseArrayDOK{T,ndims(a)}(a)

function SparseArrayDOK{T,N}(a::AbstractArray) where {T,N}
  return SparseArrayInterface.sparse_convert(SparseArrayDOK{T,N}, a)
end

function Base.resize!(a::SparseArrayDOK{<:Any,N}, new_size::NTuple{N,Integer}) where {N}
  a.dims = new_size
  return a
end

function setindex_maybe_grow!(
  a::SparseArrayDOK{<:Any,N}, value, i1::Int, I::Int...
) where {N}
  index = (i1, I...)
  if any(index .> size(a))
    resize!(a, max.(index, size(a)))
  end
  a[index...] = value
  return a
end

macro maybe_grow(ex)
  arr_name = esc(ex.args[1].args[1])
  index = esc(ex.args[1].args[2:end])
  value = esc(ex.args[2])
  quote
    SparseArrayDOKs.setindex_maybe_grow!($arr_name, $value, $index...)
  end
end
