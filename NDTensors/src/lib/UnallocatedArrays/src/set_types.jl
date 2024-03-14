using .TypeParameterAccessors: TypeParameterAccessors
using NDTensors.UnspecifiedTypes: UnspecifiedArray, UnspecifiedNumber, UnspecifiedZero

function TypeParameterAccessors.default_type_parameters(::Type{<:UnallocatedArray})
  return (
    UnspecifiedNumber{UnspecifiedZero},
    0,
    Tuple{},
    UnspecifiedArray{UnspecifiedNumber{UnspecifiedZero},0},
  )
end

unspecify_parameters(::Type{<:UnallocatedFill}) = UnallocatedFill
unspecify_parameters(::Type{<:UnallocatedZeros}) = UnallocatedZeros
