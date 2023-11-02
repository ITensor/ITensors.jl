function Base.getindex(E::Exposed{<:CuArray})
  return CUDA.@allowscalar unexpose(E)[]
end

function setindex!(E::Exposed{<:CuArray}, x::Number)
  CUDA.@allowscalar unexpose(E)[] = x
  return unexpose(E)
end

function Base.getindex(E::Exposed{<:CuArray,<:Adjoint}, I...)
  Ap = parent(E)
  return expose(Ap)[I...]
end

function Base.copy(E::Exposed{<:CuArray,<:Base.ReshapedArray})
  Ap = parent(E)
  return copy(expose(Ap))
end

Base.any(f, E::Exposed{<:CuArray,<:NDTensors.Tensor}) = any(f, data(unexpose(E)))

function Base.print_array(io::IO, E::Exposed{<:CuArray})
  return Base.print_array(io, expose(NDTensors.cpu(E)))
end
