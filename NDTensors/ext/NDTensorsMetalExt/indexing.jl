function Base.getindex(E::Exposed{<:MtlArray})
  return Metal.@allowscalar unexpose(E)[]
end

function Base.setindex!(E::Exposed{<:MtlArray}, x::Number)
  Metal.@allowscalar unexpose(E)[] = x
  return unexpose(E)
end

# Shared with `CuArray`. Move to `NDTensorsGPUArraysCoreExt`?
function Base.getindex(E::Exposed{<:MtlArray,<:Adjoint}, i, j)
  return (expose(parent(E))[j, i])'
end
