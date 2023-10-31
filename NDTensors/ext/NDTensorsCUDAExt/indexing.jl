function getindex(E::Exposed{<:CuArray})
  return CUDA.@allowscalar getindex(unexpose(E))
end

function setindex!(E::Exposed{<:CuArray}, x::Number)
  CUDA.@allowscalar setindex!(unexpose(E), x)
  return unexpose(E)
end
