using LinearAlgebra: norm
using NDTensors.SparseArrayInterface: SparseArrayInterface
using Test: @test
struct SparseArray{T,N} <: AbstractArray{T,N}
  data::Vector{T}
  dims::Tuple{Vararg{Int,N}}
  index_to_dataindex::Dict{CartesianIndex{N},Int}
  dataindex_to_index::Vector{CartesianIndex{N}}
end
function SparseArray{T,N}(dims::Tuple{Vararg{Int,N}}) where {T,N}
  return SparseArray{T,N}(
    T[], dims, Dict{CartesianIndex{N},Int}(), Vector{CartesianIndex{N}}()
  )
end
SparseArray{T,N}(dims::Vararg{Int,N}) where {T,N} = SparseArray{T,N}(dims)
SparseArray{T}(dims::Tuple{Vararg{Int}}) where {T} = SparseArray{T,length(dims)}(dims)
SparseArray{T}(dims::Vararg{Int}) where {T} = SparseArray{T}(dims)

# AbstractArray interface
Base.size(a::SparseArray) = a.dims
function Base.getindex(a::SparseArray, I...)
  return SparseArrayInterface.sparse_getindex(a, I...)
end
function Base.setindex!(a::SparseArray, I...)
  return SparseArrayInterface.sparse_setindex!(a, I...)
end
function Base.similar(a::SparseArray, elt::Type, dims::Tuple{Vararg{Int}})
  return SparseArray{elt}(dims)
end

# Minimal interface
SparseArrayInterface.nonzeros(a::SparseArray) = a.data
function SparseArrayInterface.index_to_nonzero_index(
  a::SparseArray{<:Any,N}, I::CartesianIndex{N}
) where {N}
  return get(a.index_to_dataindex, I, nothing)
end
SparseArrayInterface.nonzero_index_to_index(a::SparseArray, I) = a.dataindex_to_index[I]
function SparseArrayInterface.setindex_zero!(
  a::SparseArray{<:Any,N}, value, I::CartesianIndex{N}
) where {N}
  push!(a.data, value)
  push!(a.dataindex_to_index, I)
  a.index_to_dataindex[I] = length(a.data)
  return a
end

# Base
function Base.zero(a::SparseArray)
  return SparseArrayInterface.sparse_zero(a)
end

# Broadcasting
function Broadcast.BroadcastStyle(arraytype::Type{<:SparseArray})
  return SparseArrayInterface.SparseArrayStyle{ndims(arraytype)}()
end

# Map
function Base.map!(f, dest::AbstractArray, src::SparseArray)
  SparseArrayInterface.sparse_map!(f, dest, src)
  return dest
end
function Base.copy!(dest::AbstractArray, src::SparseArray)
  SparseArrayInterface.sparse_copy!(dest, src)
  return dest
end
function Base.copyto!(dest::AbstractArray, src::SparseArray)
  SparseArrayInterface.sparse_copyto!(dest, src)
  return dest
end
function Base.permutedims!(dest::AbstractArray, src::SparseArray, perm)
  SparseArrayInterface.sparse_permutedims!(dest, src, perm)
  return dest
end

elt = Float64

# Test
a = SparseArray{elt}(2, 3)
@test size(a) == (2, 3)
@test axes(a) == (1:2, 1:3)
@test SparseArrayInterface.nonzeros(a) == elt[]
@test iszero(SparseArrayInterface.nonzero_length(a))
@test collect(SparseArrayInterface.nonzero_indices(a)) == CartesianIndex{2}[]
@test iszero(a)
@test iszero(norm(a))
for I in eachindex(a)
  @test iszero(a)
end

a[1, 2] = 12
@test size(a) == (2, 3)
@test axes(a) == (1:2, 1:3)
@test SparseArrayInterface.nonzeros(a) == elt[12]
@test isone(SparseArrayInterface.nonzero_length(a))
@test collect(SparseArrayInterface.nonzero_indices(a)) == [CartesianIndex(1, 2)]
@test !iszero(a)
@test !iszero(norm(a))
for I in eachindex(a)
  if I == CartesianIndex(1, 2)
    @test a[I] == 12
  else
    @test iszero(a[I])
  end
end
