nonzero_length(a::AbstractArray) = length(nonzeros(a))
function nonzero_indices(a::AbstractArray)
  return Iterators.map(Inz -> nonzero_index_to_index(a, Inz), eachindex(nonzeros(a)))
end

# Derived
function index_to_nonzero_index(a::AbstractArray{<:Any,N}, I::Vararg{Int,N}) where {N}
  return index_to_nonzero_index(a, CartesianIndex(I))
end

# Helper type for constructing zero values
struct GetIndexZero end
(::GetIndexZero)(a::AbstractArray, I) = zero(eltype(a))

function getindex_nonzero(a::AbstractArray, Inz)
  return nonzeros(a)[Inz]
end

function sparse_getindex(a::AbstractArray, I...)
  return sparse_getindex(a, CartesianIndex(to_indices(a, I)))
end

function sparse_getindex(a::AbstractArray{<:Any,N}, I::Vararg{Int,N}) where {N}
  return sparse_getindex(a, CartesianIndex(I))
end

function sparse_getindex(a::AbstractArray{<:Any,N}, I::CartesianIndex{N}) where {N}
  Inz = index_to_nonzero_index(a, I)
  if isnothing(Inz)
    return getindex_zero(a, I)
  end
  return getindex_nonzero(a, Inz)
end

# Update a nonzero value
function setindex_nonzero!(a::AbstractArray, value, Inz)
  nonzeros(a)[Inz] = value
  return a
end

function sparse_setindex!(a::AbstractArray{<:Any,N}, value, I::Vararg{Int,N}) where {N}
  sparse_setindex!(a, value, CartesianIndex(I))
  return a
end

function sparse_setindex!(a::AbstractArray{<:Any,N}, value, I::CartesianIndex{N}) where {N}
  Inz = index_to_nonzero_index(a, I)
  if isnothing(Inz)
    if !iszero(value)
      # Only try setting if nonzero
      setindex_zero!(a, value, I)
    end
  else
    setindex_nonzero!(a, value, Inz)
  end
  return a
end
