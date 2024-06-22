# TypeParameterAccessors definitions
using NDTensors.TypeParameterAccessors: TypeParameterAccessors, Position
using JLArrays: JLArray

function TypeParameterAccessors.default_type_parameters(::Type{<:JLArray})
  return (Float64, 1)
end
