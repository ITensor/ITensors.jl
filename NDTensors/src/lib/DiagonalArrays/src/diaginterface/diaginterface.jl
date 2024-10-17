using Compat: allequal

diaglength(a::AbstractArray{<:Any,0}) = 1

function diaglength(a::AbstractArray)
  return minimum(size(a))
end

function isdiagindex(a::AbstractArray{<:Any,N}, I::CartesianIndex{N}) where {N}
  @boundscheck checkbounds(a, I)
  return allequal(Tuple(I))
end

function diagstride(a::AbstractArray)
  s = 1
  p = 1
  for i in 1:(ndims(a)-1)
    p *= size(a, i)
    s += p
  end
  return s
end

function diagindices(a::AbstractArray)
  maxdiag = LinearIndices(a)[CartesianIndex(ntuple(Returns(diaglength(a)), ndims(a)))]
  return 1:diagstride(a):maxdiag
end

function diagindices(a::AbstractArray{<:Any,0})
  return Base.OneTo(1)
end

function diagview(a::AbstractArray)
  return @view a[diagindices(a)]
end

function getdiagindex(a::AbstractArray, i::Integer)
  return diagview(a)[i]
end

function setdiagindex!(a::AbstractArray, v, i::Integer)
  diagview(a)[i] = v
  return a
end

function getdiagindices(a::AbstractArray, I)
  return @view diagview(a)[I]
end

function getdiagindices(a::AbstractArray, I::Colon)
  return diagview(a)
end

function setdiagindices!(a::AbstractArray, v, i::Colon)
  diagview(a) .= v
  return a
end
