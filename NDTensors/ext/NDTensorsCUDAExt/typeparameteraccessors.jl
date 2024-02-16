# # `TypeParameterAccessors.jl` overloads.
using NDTensors.TypeParameterAccessors: TypeParameterAccessors, Position, parameter
## TODO this seems like a `GPUArrays` generic function
storagemode(t::Type{<:CuArray}) = parameter(t, storagemode)
function TypeParameterAccessors.position(::Type{<:CuArray}, ::typeof(storagemode))
  return Position(3)
end

function TypeParameterAccessors.default_parameter(::Type{<:CuArray}, ::typeof(storagemode))
  return Mem.DeviceBuffer
end

## TODO this seems like a `GPUArrays` generic function
function set_storagemode(arraytype::Type{<:CuArray}, param)
  return set_parameter(arraytype, storagemode, param)
end
