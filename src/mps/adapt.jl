adapt_structure(to, x::Union{MPS,MPO}) = map(xᵢ -> adapt(to, xᵢ), x)

function NDTensors.cu(x::Union{MPS,MPO}; unified=false)
  return map(xᵢ -> NDTensors.cu(xᵢ; unified=unified), x)
end
