# Convert to an offset along the diagonal.
# Otherwise, return `nothing`.
function diagindex(a::AbstractArray{T,N}, I::CartesianIndex{N}) where {T,N}
  !allequal(Tuple(I)) && return nothing
  return first(Tuple(I))
end

function diagindex(a::AbstractArray{T,N}, I::Vararg{Int,N}) where {T,N}
  return diagindex(a, CartesianIndex(I))
end

function getdiagindex(a::AbstractArray, i::Integer)
  return diagview(a)[i]
end

function setdiagindex!(a::AbstractArray, v, i::Integer)
  diagview(a)[i] = v
  return a
end

function setdiag!(a::AbstractArray, v)
  copyto!(diagview(a), v)
  return a
end

function diaglength(a::AbstractArray)
  # length(diagview(a))
  return minimum(size(a))
end

function diagstride(A::AbstractArray)
  s = 1  
  p = 1  
  for i in 1:(ndims(A) - 1)
    p *= size(A, i)
    s += p 
  end
  return s
end

function diagindices(A::AbstractArray)
  diaglength = minimum(size(A))
  maxdiag = LinearIndices(A)[CartesianIndex(ntuple(Returns(diaglength), ndims(A)))]
  return 1:diagstride(A):maxdiag
end

function diagview(A::AbstractArray)
  return @view A[diagindices(A)]
end

function diagcopyto!(dest::AbstractArray, src::AbstractArray)
  copyto!(diagview(dest), diagview(src))
  return dest
end
