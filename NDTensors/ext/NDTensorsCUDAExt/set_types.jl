# TypeParameterAccessors definitions
using NDTensors.TypeParameterAccessors: TypeParameterAccessors, Position, set_type_parameter
using NDTensors.GPUArraysCoreExtensions: storagemode
## TODO remove TypeParameterAccessors when SetParameters is removed
function TypeParameterAccessors.position(::Type{<:CuArray}, ::typeof(eltype))
  return Position(1)
end
function TypeParameterAccessors.position(::Type{<:CuArray}, ::typeof(Base.ndims))
  return Position(2)
end
function TypeParameterAccessors.position(::Type{<:CuArray}, ::typeof(storagemode))
  return Position(3)
end

function TypeParameterAccessors.default_type_parameters(::Type{<:CuArray})
  return (Float64, 1, CUDA.Mem.DeviceBuffer)
end

function TypeParameterAccessors.set_ndims(type::Type{<:CuArray}, param)
  return set_type_parameter(type, ndims, param)
end