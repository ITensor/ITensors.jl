using FillArrays: AbstractFill
using NDTensors.TypeParameterAccessors: parameter, set_parameter
## Here are functions specifically defined for UnallocatedArrays
## not implemented by FillArrays
## TODO this might need a more generic name maybe like compute unit
function alloctype(A::AbstractFill)
  return A.alloc
end

## TODO this fails if the parameter is a type
function alloctype(Atype::Type{<:AbstractFill})
  return parameter(Atype, 4)
end

set_axestype(T::Type{<:AbstractFill}, ax::Type) = set_parameter(T, 3, ax)
