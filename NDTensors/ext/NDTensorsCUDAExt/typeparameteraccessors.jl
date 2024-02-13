# # `TypeParameterAccessors.jl` overloads.
using NDTensors.TypeParameterAccessors: Position, parameter
## TODO this seems like a `GPUArrays` generic function
storagemode(t::Type{<:CuArray}) = parameter(t, 3)
function NDTensors.TypeParameterAccessors.position(::Type{<:CuArray}, ::typeof(storagemode))
  return Position(3)
end

function NDTensors.TypeParameterAccessors.default_parameter(
  ::Type{<:CuArray}, ::typeof(eltype)
)
  return Float64
end
NDTensors.TypeParameterAccessors.default_parameter(::Type{<:CuArray}, ::typeof(ndims)) = 1
function NDTensors.TypeParameterAccessors.default_parameter(
  ::Type{<:CuArray}, ::typeof(storagemode)
)
  return Mem.DeviceBuffer
end

## TODO this seems like a `GPUArrays` generic function
function set_storagemode(arraytype::Type{<:CuArray}, store)
  return set_parameter(arraytype, storagemode, store)
end
