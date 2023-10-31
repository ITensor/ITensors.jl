function Base.getindex(E::Exposed{<:CuArray})
  return CUDA.@allowscalar getindex(unexpose(E))
end

function setindex!(E::Exposed{<:CuArray}, x::Number)
  CUDA.@allowscalar setindex!(unexpose(E), x)
  return unexpose(E)
end

function Base.getindex(E::Exposed{<:CuArray, <:Adjoint}, I...)
  Ep = parent(E)
  return Base.getindex(Ep, I...)
end

function Base.copy(E::Exposed{<:CuArray, <:Base.ReshapedArray})
  Ep = parent(E)
  return Base.copy(Ep)
end