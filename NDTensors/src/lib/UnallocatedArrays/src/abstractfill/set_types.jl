using FillArrays: AbstractFill, Fill, Zeros
using NDTensors.SetParameters: SetParameters, Position
using NDTensors.UnspecifiedTypes: UnspecifiedZero
## TODO make unit tests for all of these functions
# `SetParameters.jl` overloads.
SetParameters.get_parameter(::Type{<:AbstractFill{P1}}, ::Position{1}) where {P1} = P1
SetParameters.get_parameter(::Type{<:AbstractFill{<:Any,P2}}, ::Position{2}) where {P2} = P2
function SetParameters.get_parameter(
  ::Type{<:AbstractFill{<:Any,<:Any,P3}}, ::Position{3}
) where {P3}
  return P3
end

## Setting paramaters
# right now I am just defining the necessary ones for my implementation still working on full implementation
# Set parameter 1
function SetParameters.set_parameter(T::Type{<:AbstractFill}, ::Position{1}, P1)
  return unspecify_parameters(T){P1}
end
function SetParameters.set_parameter(
  T::Type{<:AbstractFill{<:Any,P2}}, ::Position{1}, P1
) where {P2}
  return unspecify_parameters(T){P1,P2}
end
function SetParameters.set_parameter(
  T::Type{<:AbstractFill{<:Any,P2,P3}}, ::Position{1}, P1
) where {P2,P3}
  return unspecify_parameters(T){P1,P2,P3}
end

# Set parameter 2
function SetParameters.set_parameter(
  T::Type{<:AbstractFill{P1}}, ::Position{2}, P2
) where {P1}
  return unspecify_parameters(T){P1,P2}
end
function SetParameters.set_parameter(
  T::Type{<:AbstractFill{P1,P}}, ::Position{2}, P2
) where {P1,P}
  return unspecify_parameters(T){P1,P2}
end
function SetParameters.set_parameter(
  T::Type{<:AbstractFill{P1,P,P3}}, ::Position{2}, P2
) where {P1,P,P3}
  return unspecify_parameters(T){P1,P2,P3}
end

# Set parameter 3
function SetParameters.set_parameter(
  T::Type{<:AbstractFill{P1}}, ::Position{3}, P3
) where {P1}
  return unspecify_parameters(T){P1,<:Any,P3}
end
function SetParameters.set_parameter(
  T::Type{<:AbstractFill{P1,P2}}, ::Position{3}, P3
) where {P1,P2}
  return unspecify_parameters(T){P1,P2,P3}
end
function SetParameters.set_parameter(
  T::Type{<:AbstractFill{P1,P2,P}}, ::Position{3}, P3
) where {P1,P2,P}
  return unspecify_parameters(T){P1,P2,P3}
end

## default parameters
function SetParameters.default_parameter(::Type{<:AbstractFill}, ::Position{1})
  return UnspecifiedTypes.UnspecifiedZero
end
SetParameters.default_parameter(::Type{<:AbstractFill}, ::Position{2}) = 0
SetParameters.default_parameter(::Type{<:AbstractFill}, ::Position{3}) = Tuple{}

SetParameters.nparameters(::Type{<:AbstractFill}) = Val(3)

## These helper functions take a UnallocatedArray type and 
## remove all the parameters, this way all parameters can be set
## at once in the `set_parameter` functions above.
unspecify_parameters(::Type{<:Fill}) = Fill
unspecify_parameters(::Type{<:Zeros}) = Zeros
