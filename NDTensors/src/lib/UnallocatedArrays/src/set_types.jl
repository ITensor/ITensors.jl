using NDTensors.TypeParameterAccessors: TypeParameterAccessors
using NDTensors.UnspecifiedTypes: UnspecifiedArray, UnspecifiedNumber, UnspecifiedZero

## TODO set ndims position.
# ## default parameters
function TypeParameterAccessors.default_parameter(::Type{<:UnallocatedArray}, ::Position{4})
  return UnspecifiedArray{UnspecifiedNumber{UnspecifiedZero},0}
end

TypeParameterAccessors.nparameters(::Type{<:UnallocatedArray}) = Val(4)

unspecify_parameters(::Type{<:UnallocatedFill}) = UnallocatedFill
unspecify_parameters(::Type{<:UnallocatedZeros}) = UnallocatedZeros
