module SparseArrayDOKs
using Dictionaries: Dictionary, set!
using SparseArrays: SparseArrays, AbstractSparseArray

# Also look into:
# https://juliaarrays.github.io/ArrayInterface.jl/stable/sparsearrays/

# Required `SparseArrayInterface` interface.
# https://github.com/Jutho/SparseArrayKit.jl interface functions
nonzero_keys(a::AbstractArray) = error("Not implemented")
nonzero_values(a::AbstractArray) = error("Not implemented")
nonzero_pairs(a::AbstractArray) = error("Not implemented")

# A dictionary-like structure
# TODO: Rename `nonzeros`, `structural_nonzeros`, etc.?
nonzero_structure(a::AbstractArray) = error("Not implemented")

# Derived `SparseArrayInterface` interface.
nonzero_length(a::AbstractArray) = length(nonzero_keys(a))
is_structural_nonzero(a::AbstractArray, I) = I ∈ nonzero_keys(a)

# Overload if zero value is index dependent or
# doesn't match element type.
zerotype(a::AbstractArray) = eltype(a)
getindex_nonzero(a::AbstractArray, I) = nonzero_structure(a)[I]
getindex_zero(a::AbstractArray, I) = zero(zerotype(a))
function setindex_zero!(a::AbstractArray, value, I)
  # TODO: This may need to be modified.
  nonzero_structure(a)[I] = value
  return a
end
function setindex_nonzero!(a::AbstractArray, value, I)
  nonzero_structure(a)[I] = value
  return a
end

struct Zero
end
(::Zero)(type, I) = zero(type)

default_zero_type(type::Type) = type
default_zero() = Zero() # (eltype, I) -> zero(eltype)
default_keytype(ndims::Int) = CartesianIndex{ndims}
default_data(type::Type, ndims::Int) = Dictionary{default_keytype(ndims),type}()

# TODO: Define a constructor with a default `zero`.
struct SparseArrayDOK{T,N,ZT,Zero} <: AbstractSparseArray{T,Int,N}
  data::Dictionary{CartesianIndex{N},T}
  dims::NTuple{N,Int}
  zero::Zero
end

# Constructors
function SparseArrayDOK{T,N,ZT,Zero}(dims::Tuple{Vararg{Int}}, zero) where {T,N,ZT,Zero}
  return SparseArrayDOK{T,N,ZT,Zero}(default_data(T, N), dims, zero)
end

function SparseArrayDOK{T,N,ZT}(dims::Tuple{Vararg{Int}}, zero) where {T,N,ZT}
  return SparseArrayDOK{T,N,ZT,typeof(zero)}(dims, default_zero())
end

function SparseArrayDOK{T,N,ZT}(dims::Tuple{Vararg{Int}}) where {T,N,ZT}
  return SparseArrayDOK{T,N,ZT}(dims, default_zero())
end

function SparseArrayDOK{T,N}(dims::Tuple{Vararg{Int}}) where {T,N}
  return SparseArrayDOK{T,N,default_zero_type(T)}(dims)
end

function SparseArrayDOK{T}(dims::Tuple{Vararg{Int}}) where {T}
  return SparseArrayDOK{T,length(dims)}(dims)
end

function SparseArrayDOK{T}(dims::Int...) where {T}
  return SparseArrayDOK{T}(dims)
end

# undef
function SparseArrayDOK{T,N,ZT,Zero}(::UndefInitializer, dims::Tuple{Vararg{Int}}, zero) where {T,N,ZT,Zero}
  return SparseArrayDOK{T,N,ZT,Zero}(dims, zero)
end

function SparseArrayDOK{T,N,ZT}(::UndefInitializer, dims::Tuple{Vararg{Int}}, zero) where {T,N,ZT}
  return SparseArrayDOK{T,N,ZT}(dims, zero)
end

function SparseArrayDOK{T,N,ZT}(::UndefInitializer, dims::Tuple{Vararg{Int}}) where {T,N,ZT}
  return SparseArrayDOK{T,N,ZT}(dims)
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

# Required `SparseArrayInterface` interface
nonzero_structure(a::SparseArrayDOK) = a.data
# TODO: Make this a generic function.
nonzero_keys(a::SparseArrayDOK) = keys(nonzero_structure(a))

# Julia Base `AbstractSparseArray` interface
SparseArrays.nnz(a::SparseArrayDOK) = nonzero_length(a)

# Optional SparseArray interface
# TODO: Use `SetParameters`.
zerotype(a::SparseArrayDOK{<:Any,<:Any,ZT}) where {ZT} = ZT
zero_value(a::SparseArrayDOK, I) = a.zero(zerotype(a), I)

# Accessors
Base.size(a::SparseArrayDOK) = a.dims

function Base.getindex(a::SparseArrayDOK{<:Any,N}, I::Vararg{Int,N}) where {N}
  return a[CartesianIndex(I)]
end

function Base.getindex(a::SparseArrayDOK{<:Any,N}, I::CartesianIndex{N}) where {N}
  if !is_structural_nonzero(a, I)
    return getindex_zero(a, I)
  end
  return getindex_nonzero(a, I)
end

# `SparseArrayInterface` interface
function setindex_zero!(a::SparseArrayDOK, value, I)
  # TODO: This is specific to the `Dictionaries.jl`
  # interface, make more generic?
  set!(nonzero_structure(a), I, value)
  return a
end


function Base.setindex!(a::SparseArrayDOK{<:Any,N}, value, I::Vararg{Int,N}) where {N}
  a[CartesianIndex(I)] = value
  return a
end

function Base.setindex!(a::SparseArrayDOK{<:Any,N}, value, I::CartesianIndex{N}) where {N}
  if !is_structural_nonzero(a, I)
    setindex_zero!(a, value, I)
  end
  setindex_nonzero!(a, value, I)
  return a
end

# similar
# TODO: How does this deal with the converting the zero type?
Base.similar(a::SparseArrayDOK{T,N,ZT,Zero}) where {T,N,ZT,Zero} = SparseArrayDOK{T,N,ZT,Zero}(undef, size(a), a.zero)

end
