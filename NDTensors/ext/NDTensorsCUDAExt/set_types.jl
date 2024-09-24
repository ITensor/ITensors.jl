# TypeParameterAccessors definitions
using CUDA: CUDA, CuArray
using NDTensors.TypeParameterAccessors:
  TypeParameterAccessors, Position, default_type_parameters
using NDTensors.GPUArraysCoreExtensions: storagemode

function TypeParameterAccessors.position(::Type{<:CuArray}, ::typeof(storagemode))
  return Position(3)
end

function TypeParameterAccessors.default_type_parameters(::Type{<:CuArray})
  return (default_type_parameters(AbstractArray)..., CUDA.Mem.DeviceBuffer)
end
