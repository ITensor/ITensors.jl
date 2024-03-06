using FillArrays: AbstractFill
using NDTensors.TypeParameterAccessors: TypeParameterAccessors, Position, type_parameter, set_type_parameters
## Here are functions specifically defined for UnallocatedArrays
## not implemented by FillArrays
## TODO this might need a more generic name maybe like compute unit
function alloctype(A::AbstractFill)
  return A.alloc
end

alloctype(Atype::Type{<:AbstractFill}) = type_parameter(Atype, alloctype)
axestype(Atype::Type{<:AbstractFill}) = type_parameter(Atype, axestype)
TypeParameterAccessors.parameter(::Type{<:AbstractFill}, ::typeof(axestype)) = Position(3)
TypeParameterAccessors.parameter(::Type{<:AbstractFill}, ::typeof(alloctype)) = Position(4)
## TODO this fails if the parameter is a type

set_axestype(T::Type{<:AbstractFill}, ax::Type) = set_type_parameters(T,  axestype, ax)
