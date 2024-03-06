using NDTensors.TypeParameterAccessors: TypeParameterAccessors, Position
using NDTensors.UnspecifiedTypes: UnspecifiedArray, UnspecifiedNumber, UnspecifiedZero

TypeParameterAccessors.parameter(::Type{<:AbstractFill}, ::typeof(axestype)) = Position(3)
TypeParameterAccessors.parameter(::Type{<:AbstractFill}, ::typeof(alloctype)) = Position(4)
## TODO this is broken and need to define for UnallocatedArray
function TypeParameterAccessors.default_type_parameters(::Type{<:AbstractFill})
  return (Float64, 0, Tuple{})
end

# ## default parameters
# function SetParameters.default_parameter(::Type{<:UnallocatedArray}, ::Position{4})
#   return UnspecifiedArray{UnspecifiedNumber{UnspecifiedZero},0}
# end
