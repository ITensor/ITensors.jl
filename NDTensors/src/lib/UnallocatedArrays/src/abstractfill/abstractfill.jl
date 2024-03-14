using FillArrays: AbstractFill
using NDTensors.TypeParameterAccessors: TypeParameterAccessors, Position, type_parameter, set_type_parameter
## Here are functions specifically defined for UnallocatedArrays
## not implemented by FillArrays
## TODO this might need a more generic name maybe like compute unit
function alloctype(A::AbstractFill)
  return A.alloc
end

## TODO this fails if the parameter is a type
function alloctype(Atype::Type{<:AbstractFill})
  return type_parameter(Atype, alloctype)
end

axestype(T::Type{<:AbstractArray}) = type_parameter(axestype)
set_axestype(T::Type{<:AbstractFill}, ax::Type) = s(T, axestype, ax)

TypeParameterAccessors.position(::Type{<:AbstractFill}, ::typeof(alloctype)) = Position(4)
TypeParameterAccessors.position(::Type{<:AbstractFill}, ::typeof(axestype)) = Position(3)
TypeParameterAccessors.default_type_parameters(::Type{<:AbstractFill}) = (Float64, 0, Tuple{})
