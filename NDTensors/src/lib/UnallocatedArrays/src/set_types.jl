using .TypeParameterAccessors: TypeParameterAccessors
using NDTensors.UnspecifiedTypes: UnspecifiedArray, UnspecifiedNumber, UnspecifiedZero

TypeParameterAccessors.default_type_parameters(::Type{<:UnallocatedArray}) = (UnspecifiedNumber{UnspecifiedZero}, 0, Tuple{}, UnspecifiedArray{UnspecifiedNumber{UnspecifiedZero},0})

unspecify_parameters(::Type{<:UnallocatedFill}) = UnallocatedFill
unspecify_parameters(::Type{<:UnallocatedZeros}) = UnallocatedZeros
