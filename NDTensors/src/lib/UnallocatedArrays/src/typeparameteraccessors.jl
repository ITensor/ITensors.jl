using FillArrays: AbstractFill
using NDTensors.TypeParameterAccessors: TypeParameterAccessors
using NDTensors.UnspecifiedTypes: UnspecifiedArray, UnspecifiedNumber, UnspecifiedZero

# ## default parameters
function TypeParameterAccessors.default_parameter(
  ::Type{<:UnallocatedArray}, ::typeof(alloctype)
)
  return UnspecifiedArray{UnspecifiedNumber{UnspecifiedZero},0}
end

function TypeParameterAccessors.default_parameter(
  ::Type{<:AbstractFill}, ::typeof(axestype)
)
  return Tuple{}
end

TypeParameterAccessors.parameter_name(::Type{<:AbstractFill}, ::Position{3}) = axestype
function TypeParameterAccessors.parameter_name(
  ::Type{<:UnallocatedArray}, ::Position{4}
)
  return alloctype
end
