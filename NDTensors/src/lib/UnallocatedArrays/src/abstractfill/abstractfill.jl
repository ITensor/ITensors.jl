using FillArrays: AbstractFill
using NDTensors.TypeParameterAccessors:
  TypeParameterAccessors, Position, parameter, set_parameter
## Here are functions specifically defined for UnallocatedArrays
## not implemented by FillArrays
## TODO this might need a more generic name maybe like compute unit
function alloctype(A::AbstractFill)
  return A.alloc
end

## TODO this fails if the parameter is a type
function alloctype(Atype::Type{<:AbstractFill})
  return parameter(Atype, alloctype)
end

axestype(t::Type{<:AbstractFill}) = parameter(t, axestype)

TypeParameterAccessors.position(::Type{<:AbstractFill}, ::typeof(axestype)) = Position(3)
TypeParameterAccessors.position(::Type{<:AbstractFill}, ::typeof(alloctype)) = Position(4)
