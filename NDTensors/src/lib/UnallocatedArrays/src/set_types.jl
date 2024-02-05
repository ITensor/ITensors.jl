using NDTensors.TypeParameterAccessors: TypeParameterAccessors, Position
using NDTensors.UnspecifiedTypes: UnspecifiedArray, UnspecifiedNumber, UnspecifiedZero
# ## TODO make unit tests for all of these functions
## TODO All I need to do is overload AbstractFill functions with 4 parameters
# `TypeParameterAccessors.jl` overloads.
function TypeParameterAccessors.get_parameter(
  ::Type{<:UnallocatedArray{<:Any,<:Any,<:Any,P4}}, ::Position{4}
) where {P4}
  return P4
end

# ## Setting paramaters
function TypeParameterAccessors.set_parameter(
  T::Type{<:UnallocatedArray{P,P2,P3,P4}}, ::Position{1}, P1
) where {P,P2,P3,P4}
  return unspecify_parameters(T){P1,P2,P3,P4}
end

function TypeParameterAccessors.set_parameter(
  T::Type{<:UnallocatedArray{P1,P,P3,P4}}, ::Position{2}, P2
) where {P1,P,P3,P4}
  return unspecify_parameters(T){P1,P2,P3,P4}
end

function TypeParameterAccessors.set_parameter(
  T::Type{<:UnallocatedArray{P1,P2,P,P4}}, ::Position{3}, P3
) where {P1,P2,P,P4}
  return unspecify_parameters(T){P1,P2,P3,P4}
end

function TypeParameterAccessors.set_parameter(
  T::Type{<:UnallocatedArray{P1,P2,P3}}, ::Position{4}, P4
) where {P1,P2,P3}
  return unspecify_parameters(T){P1,P2,P3,P4}
end

# ## default parameters
function TypeParameterAccessors.default_parameter(::Type{<:UnallocatedArray}, ::Position{4})
  return UnspecifiedArray{UnspecifiedNumber{UnspecifiedZero},0}
end

TypeParameterAccessors.nparameters(::Type{<:UnallocatedArray}) = Val(4)

unspecify_parameters(::Type{<:UnallocatedFill}) = UnallocatedFill
unspecify_parameters(::Type{<:UnallocatedZeros}) = UnallocatedZeros
