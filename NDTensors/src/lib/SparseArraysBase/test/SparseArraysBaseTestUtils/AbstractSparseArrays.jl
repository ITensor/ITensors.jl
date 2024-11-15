module AbstractSparseArrays
using ArrayLayouts: ArrayLayouts, MatMulMatAdd, MemoryLayout, MulAdd
using NDTensors.SparseArraysBase: SparseArraysBase, AbstractSparseArray, Zero

struct SparseArray{T,N,Zero} <: AbstractSparseArray{T,N}
  data::Vector{T}
  dims::Tuple{Vararg{Int,N}}
  index_to_dataindex::Dict{CartesianIndex{N},Int}
  dataindex_to_index::Vector{CartesianIndex{N}}
  zero::Zero
end
function SparseArray{T,N}(dims::Tuple{Vararg{Int,N}}; zero=Zero()) where {T,N}
  return SparseArray{T,N,typeof(zero)}(
    T[], dims, Dict{CartesianIndex{N},Int}(), Vector{CartesianIndex{N}}(), zero
  )
end
function SparseArray{T,N}(dims::Vararg{Int,N}; kwargs...) where {T,N}
  return SparseArray{T,N}(dims; kwargs...)
end
function SparseArray{T}(dims::Tuple{Vararg{Int}}; kwargs...) where {T}
  return SparseArray{T,length(dims)}(dims; kwargs...)
end
function SparseArray{T}(::UndefInitializer, dims::Tuple{Vararg{Int}}; kwargs...) where {T}
  return SparseArray{T}(dims; kwargs...)
end
SparseArray{T}(dims::Vararg{Int}; kwargs...) where {T} = SparseArray{T}(dims; kwargs...)

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
  SparseArraysBase.sparse_mul!(a_dest, a1, a2, α, β)
  return a_dest
end

# AbstractArray interface
Base.size(a::SparseArray) = a.dims
function Base.similar(a::SparseArray, elt::Type, dims::Tuple{Vararg{Int}})
  return SparseArray{elt}(dims)
end

# Minimal interface
SparseArraysBase.getindex_zero_function(a::SparseArray) = a.zero
SparseArraysBase.sparse_storage(a::SparseArray) = a.data
function SparseArraysBase.index_to_storage_index(
  a::SparseArray{<:Any,N}, I::CartesianIndex{N}
) where {N}
  return get(a.index_to_dataindex, I, nothing)
end
SparseArraysBase.storage_index_to_index(a::SparseArray, I) = a.dataindex_to_index[I]
function SparseArraysBase.setindex_notstored!(
  a::SparseArray{<:Any,N}, value, I::CartesianIndex{N}
) where {N}
  push!(a.data, value)
  push!(a.dataindex_to_index, I)
  a.index_to_dataindex[I] = length(a.data)
  return a
end

# Empty the storage, helps with efficiency in `map!` to drop
# zeros.
function SparseArraysBase.dropall!(a::SparseArray)
  empty!(a.data)
  empty!(a.index_to_dataindex)
  empty!(a.dataindex_to_index)
  return a
end
end
