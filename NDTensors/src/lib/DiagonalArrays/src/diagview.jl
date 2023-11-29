using Compat: allequal

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

struct DiagIndex
  I::Int
end

function Base.getindex(a::AbstractArray, i::DiagIndex)
  return getdiagindex(a, i.I)
end

function Base.setindex!(a::AbstractArray, v, i::DiagIndex)
  setdiagindex!(a, v, i.I)
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

function diagstride(a::AbstractArray)
  s = 1
  p = 1
  for i in 1:(ndims(a) - 1)
    p *= size(a, i)
    s += p
  end
  return s
end

function diagindices(a::AbstractArray)
  diaglength = minimum(size(a))
  maxdiag = LinearIndices(a)[CartesianIndex(ntuple(Returns(diaglength), ndims(a)))]
  return 1:diagstride(a):maxdiag
end

function diagindices(a::AbstractArray{<:Any,0})
  return Base.OneTo(1)
end

function diagview(a::AbstractArray)
  return @view a[diagindices(a)]
end

function diagcopyto!(dest::AbstractArray, src::AbstractArray)
  copyto!(diagview(dest), diagview(src))
  return dest
end

struct DiagIndices end

function Base.getindex(a::AbstractArray, ::DiagIndices)
  return diagview(a)
end

function Base.setindex!(a::AbstractArray, v, ::DiagIndices)
  setdiag!(a, v)
  return a
end
