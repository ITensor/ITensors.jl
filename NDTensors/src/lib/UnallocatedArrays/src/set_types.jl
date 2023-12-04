# ## TODO make unit tests for all of these functions
## TODO All I need to do is overload AbstractFill functions with 4 parameters
# `SetParameters.jl` overloads.
function SetParameters.get_parameter(
  ::Type{<:UnallocatedFill{<:Any,<:Any,<:Any,P4}}, ::Position{4}
) where {P4}
  return P4
end
function SetParameters.get_parameter(
  ::Type{<:UnallocatedZeros{<:Any,<:Any,<:Any,P4}}, ::Position{4}
) where {P4}
  return P4
end

# ## Setting paramaters
function SetParameters.set_parameter(
  T::Type{<:UnallocatedFill{P1,P2,P3,<:Any}}, ::Position{4}, P4::Type{<:AbstractArray}
) where {P1,P2,P3}
  return T{P4}
end
function SetParameters.set_parameter(
  T::Type{<:UnallocatedZeros{P1,P2,P3,<:Any}}, ::Position{4}, P4::Type{<:AbstractArray}
) where {P1,P2,P3}
  return T{P4}
end

# ## default parameters
function SetParameters.default_parameter(::Type{<:UnallocatedFill}, ::Position{4})
  return UnspecifiedTypes.UnspecifiedArray
end
function SetParameters.default_parameter(::Type{<:UnallocatedZeros}, ::Position{4})
  return UnspecifiedTypes.UnspecifiedArray
end

SetParameters.nparameters(::Type{<:UnallocatedFill}) = Val(4)
SetParameters.nparameters(::Type{<:UnallocatedZeros}) = Val(4)
