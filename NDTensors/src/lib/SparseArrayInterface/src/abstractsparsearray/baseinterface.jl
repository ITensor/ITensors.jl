using ..SparseArrayInterface: SparseArrayInterface

Base.size(a::AbstractSparseArray) = error("Not implemented")

function Base.similar(a::AbstractSparseArray, elt::Type, dims::Tuple{Vararg{Int}})
  return error("Not implemented")
end

function Base.getindex(a::AbstractSparseArray, I...)
  return SparseArrayInterface.sparse_getindex(a, I...)
end

# Fixes ambiguity error with `ArrayLayouts`.
function Base.getindex(a::AbstractSparseArray, I1::AbstractVector, I2::AbstractVector)
  return SparseArrayInterface.sparse_getindex(a, I1, I2)
end

function Base.isassigned(a::AbstractSparseArray, I::Integer...)
  return SparseArrayInterface.sparse_isassigned(a, I...)
end

function Base.setindex!(a::AbstractSparseArray, I...)
  return SparseArrayInterface.sparse_setindex!(a, I...)
end
