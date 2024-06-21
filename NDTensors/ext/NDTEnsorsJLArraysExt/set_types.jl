# TypeParameterAccessors definitions
using NDTensors.TypeParameterAccessors: TypeParameterAccessors, Position
using NDTensors.GPUArraysCoreExtensions: storagemode
using JLArrays: JLArrays, JLArray

function TypeParameterAccessors.default_type_parameters(::Type{<:JLArray})
  return (Float64, 1)
end
TypeParameterAccessors.position(::Type{<:JLArray}, ::typeof(eltype)) = Position(1)
TypeParameterAccessors.position(::Type{<:JLArray}, ::typeof(ndims)) = Position(2)
