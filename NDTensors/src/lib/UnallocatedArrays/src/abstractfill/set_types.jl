using FillArrays: AbstractFill
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
  return strip_parameters(T){P1}
end
function SetParameters.set_parameter(
  T::Type{<:AbstractFill{<:Any,P2}}, ::Position{1}, P1
) where {P2}
  return strip_parameters(T){P1,P2}
end
function SetParameters.set_parameter(
  T::Type{<:AbstractFill{<:Any,P2,P3}}, ::Position{1}, P1
) where {P2,P3}
  return strip_parameters(T){P1,P2,P3}
end

# Set parameter 2
function SetParameters.set_parameter(
  T::Type{<:AbstractFill{P1}}, ::Position{2}, P2
) where {P1}
  return strip_parameters(T){P1,P2}
end
function SetParameters.set_parameter(
  T::Type{<:AbstractFill{P1,<:Any}}, ::Position{2}, P2
) where {P1}
  return strip_parameters(T){P1,P2}
end
function SetParameters.set_parameter(
  T::Type{<:AbstractFill{P1,<:Any,P3}}, ::Position{2}, P2
) where {P1,P3}
  return T{P1,P2,P3}
end

# Set parameter 3
function SetParameters.set_parameter(
  T::Type{<:AbstractFill{P1}}, ::Position{3}, P3
) where {P1}
  return strip_parameters(T){P1,<:Any,P3}
end
function SetParameters.set_parameter(
  T::Type{<:AbstractFill{P1,P2}}, ::Position{3}, P3
) where {P1,P2}
  return strip_parameters(T){P1,P2,P3}
end
function SetParameters.set_parameter(
  T::Type{<:AbstractFill{P1,P2,<:Any}}, ::Position{3}, P3
) where {P1,P2}
  return strip_parameters(T){P1,P2,P3}
end

## TODO define a specify_parameters function
## To quickly specify P1, P2, and P3 
## default parameters
function SetParameters.default_parameter(::Type{<:AbstractFill}, ::Position{1})
  return UnspecifiedTypes.UnspecifiedZero
end
SetParameters.default_parameter(::Type{<:AbstractFill}, ::Position{2}) = 0
SetParameters.default_parameter(::Type{<:AbstractFill}, ::Position{3}) = Tuple{}

SetParameters.nparameters(::Type{<:AbstractFill}) = Val(3)

## This is a helper function which can be renamed.
## Essentially, because I use AbstractFill I don't know what the name
## of the object is in `set_parameter`, so I use this to find the 
## name with out any of the parameters attached.
strip_parameters(::Type{<:Fill}) = Fill
strip_parameters(::Type{<:Zeros}) = Zeros
