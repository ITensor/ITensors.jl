using NDTensors.TypeParameterAccessors: TypeParameterAccessors
using NDTensors.UnspecifiedTypes: UnspecifiedArray, UnspecifiedNumber, UnspecifiedZero

# ## default parameters
function TypeParameterAccessors.default_parameter(
  ::Type{<:UnallocatedArray}, ::typeof(alloctype)
)
  return UnspecifiedArray{UnspecifiedNumber{UnspecifiedZero},0}
end

# ## default parameters
function TypeParameterAccessors.default_parameters(::Type{<:UnallocatedArray})
  return (eltype, ndims, axestype, alloctype)
end
