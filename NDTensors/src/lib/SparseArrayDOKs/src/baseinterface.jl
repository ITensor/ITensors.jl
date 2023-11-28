using ..SparseArrayInterface: SparseArrayInterface

Base.size(a::SparseArrayDOK) = a.dims

function Base.getindex(a::SparseArrayDOK, I...)
  return SparseArrayInterface.sparse_getindex(a, I...)
end

function Base.setindex!(a::SparseArrayDOK, I...)
  return SparseArrayInterface.sparse_setindex!(a, I...)
end

function Base.similar(a::SparseArrayDOK, elt::Type, dims::Tuple{Vararg{Int}})
  return SparseArrayDOK{elt}(undef, dims)
end
