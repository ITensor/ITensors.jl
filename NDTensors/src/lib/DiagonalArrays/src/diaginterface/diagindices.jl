# Represents a set of linear offsets along the diagonal
struct DiagIndices{I}
  i::I
end
indices(i::DiagIndices) = i.i

function Base.getindex(a::AbstractArray, I::DiagIndices)
  return getdiagindices(a, indices(I))
end

function Base.setindex!(a::AbstractArray, value, i::DiagIndices)
  setdiagindices!(a, value, indices(i))
  return a
end
