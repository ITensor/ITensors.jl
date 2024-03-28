using CUDA: CuArray
using GPUArraysCore: @allowscalar
using NDTensors: NDTensors
using NDTensors.Expose: Exposed, expose, parent, unexpose

using GPUArraysCore: @allowscalar
function Base.getindex(E::Exposed{<:CuArray})
  return @allowscalar unexpose(E)[]
end

function Base.setindex!(E::Exposed{<:CuArray}, x::Number)
  @allowscalar unexpose(E)[] = x
  return unexpose(E)
end

function Base.getindex(E::Exposed{<:CuArray,<:Adjoint}, i, j)
  return (expose(parent(E))[j, i])'
end

Base.any(f, E::Exposed{<:CuArray,<:NDTensors.Tensor}) = any(f, data(unexpose(E)))

function Base.print_array(io::IO, E::Exposed{<:CuArray})
  return Base.print_array(io, expose(NDTensors.cpu(E)))
end
