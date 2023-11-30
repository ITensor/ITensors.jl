# Represents a linear offset along the diagonal
struct DiagIndex{I}
  i::I
end
index(i::DiagIndex) = i.i

function Base.getindex(a::AbstractArray, i::DiagIndex)
  return getdiagindex(a, index(i))
end

function Base.setindex!(a::AbstractArray, value, i::DiagIndex)
  setdiagindex!(a, value, index(i))
  return a
end
