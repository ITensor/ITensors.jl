using NDTensors.TypeParameterAccessors: TypeParameterAccessors
using NDTensors.UnspecifiedTypes: UnspecifiedArray, UnspecifiedNumber, UnspecifiedZero

## TODO set ndims position.
# ## default parameters
function TypeParameterAccessors.default_parameter(
  ::Type{<:UnallocatedArray}, ::typeof(alloctype)
)
  return UnspecifiedArray{UnspecifiedNumber{UnspecifiedZero},0}
end
