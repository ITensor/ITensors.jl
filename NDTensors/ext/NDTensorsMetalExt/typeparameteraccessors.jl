# # `TypeParameterAccessors.jl` overloads.
using NDTensors.TypeParameterAccessors:
  TypeParameterAccessors, Position, parameter, position, set_parameter
## TODO this seems like a `GPUArrays` generic function
storagemode(T::Type{<:MtlArray}) = parameter(T, 3)
## TODO this seems like a `GPUArrays` generic function
TypeParameterAccessors.position(::Type{<:MtlArray}, ::typeof(storagemode)) = Position(3)

function NDTensors.TypeParameterAccessors.default_parameter(
  ::Type{<:MtlArray}, ::typeof{eltype}
)
  return Float32
end
NDTensors.TypeParameterAccessors.default_parameter(::Type{<:MtlArray}, ::typeof{ndims}) = 1
function NDTensors.TypeParameterAccessors.default_parameter(
  ::Type{<:MtlArray}, ::typeof{storagemode}
)
  return Metal.DefaultStorageMode
end

## TODO this seems like a `GPUArrays` generic function
function set_storagemode(arraytype::Type{<:MtlArray}, store)
  return set_parameter(arraytype, storagemode, store)
end