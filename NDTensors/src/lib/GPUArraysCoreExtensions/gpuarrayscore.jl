using .TypeParameterAccessors: TypeParameterAccessors, type_parameter
using GPUArraysCore: AbstractGPUArray

TypeParameterAccessors.default_type_parameters(::Type{<:AbstractGPUArray}) = (Float32, 1)

function storagemode(object)
  return storagemode(typeof(object))
end
function storagemode(type::Type)
  return type_parameter(type, storagemode)
end
