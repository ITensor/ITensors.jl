function SparseArrayDOK{T,N,Zero}(a::SparseArrayDOK{T,N,Zero}) where {T,N,Zero}
  return copy(a)
end

function Base.convert(
  ::Type{SparseArrayDOK{T,N,Zero}}, a::SparseArrayDOK{T,N,Zero}
) where {T,N,Zero}
  return a
end

Base.convert(type::Type{<:SparseArrayDOK}, a::AbstractArray) = type(a)

SparseArrayDOK(a::AbstractArray) = SparseArrayDOK{eltype(a)}(a)

SparseArrayDOK{T}(a::AbstractArray) where {T} = SparseArrayDOK{T,ndims(a)}(a)

function SparseArrayDOK{T,N}(a::AbstractArray) where {T,N}
  return SparseArrayInterface.sparse_convert(SparseArrayDOK{T,N}, a)
end
