module AbstractSparseArrays
using ArrayLayouts: ArrayLayouts, MatMulMatAdd, MemoryLayout, MulAdd
using NDTensors.SparseArrayInterface: SparseArrayInterface, AbstractSparseArray

struct SparseArray{T,N} <: AbstractSparseArray{T,N}
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
function SparseArray{T}(::UndefInitializer, dims::Tuple{Vararg{Int}}) where {T}
  return SparseArray{T}(dims)
end
SparseArray{T}(dims::Vararg{Int}) where {T} = SparseArray{T}(dims)

# ArrayLayouts interface
struct SparseLayout <: MemoryLayout end
ArrayLayouts.MemoryLayout(::Type{<:SparseArray}) = SparseLayout()
function Base.similar(::MulAdd{<:SparseLayout,<:SparseLayout}, elt::Type, axes)
  return similar(SparseArray{elt}, axes)
end
function ArrayLayouts.materialize!(
  m::MatMulMatAdd{<:SparseLayout,<:SparseLayout,<:SparseLayout}
)
  α, a1, a2, β, a_dest = m.α, m.A, m.B, m.β, m.C
  SparseArrayInterface.sparse_mul!(a_dest, a1, a2, α, β)
  return a_dest
end

# AbstractArray interface
Base.size(a::SparseArray) = a.dims
function Base.similar(a::SparseArray, elt::Type, dims::Tuple{Vararg{Int}})
  return SparseArray{elt}(dims)
end

# Minimal interface
SparseArrayInterface.sparse_storage(a::SparseArray) = a.data
function SparseArrayInterface.index_to_storage_index(
  a::SparseArray{<:Any,N}, I::CartesianIndex{N}
) where {N}
  return get(a.index_to_dataindex, I, nothing)
end
SparseArrayInterface.storage_index_to_index(a::SparseArray, I) = a.dataindex_to_index[I]
function SparseArrayInterface.setindex_notstored!(
  a::SparseArray{<:Any,N}, value, I::CartesianIndex{N}
) where {N}
  push!(a.data, value)
  push!(a.dataindex_to_index, I)
  a.index_to_dataindex[I] = length(a.data)
  return a
end

# Empty the storage, helps with efficiency in `map!` to drop
# zeros.
function SparseArrayInterface.dropall!(a::SparseArray)
  empty!(a.data)
  empty!(a.index_to_dataindex)
  empty!(a.dataindex_to_index)
  return a
end
end
