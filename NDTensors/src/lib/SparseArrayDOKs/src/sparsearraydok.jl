using Dictionaries: Dictionary
using ..SparseArrayInterface:
  SparseArrayInterface, AbstractSparseArray, getindex_zero_function

# TODO: Parametrize by `data`?
struct SparseArrayDOK{T,N,Zero} <: AbstractSparseArray{T,N}
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

function SparseArrayDOK{T}(::UndefInitializer, dims::Int...) where {T}
  return SparseArrayDOK{T}(dims...)
end

function SparseArrayDOK{T}(::UndefInitializer, dims::Tuple{Vararg{Int}}, zero) where {T}
  return SparseArrayDOK{T}(dims, zero)
end

# Base `AbstractArray` interface
Base.size(a::SparseArrayDOK) = a.dims

SparseArrayInterface.getindex_zero_function(a::SparseArrayDOK) = a.zero

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
