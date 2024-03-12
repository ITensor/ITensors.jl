using .TypeParameterAccessors: TypeParameterAccessors, type_parameter
using GPUArraysCore: AbstractGPUArray
using Adapt: Adapt

TypeParameterAccessors.default_type_parameters(::Type{<:AbstractGPUArray}) = (Float32, 1)

function storagemode(array::AbstractGPUArray)
  return storagemode(typeof(array))
end
function storagemode(type::Type{<:AbstractGPUArray})
  return type_parameter(type, storagemode)
end
