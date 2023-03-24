adapt_structure(to, x::Union{MPS,MPO}) = map(xᵢ -> adapt(to, xᵢ), x)

NDTensors.cu(x::Union{MPS, MPO}; unified = false) = map(xᵢ -> NDTensors.cu(xᵢ, unified = unified), x)