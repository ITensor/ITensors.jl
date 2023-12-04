## TODO make unit tests for all of these functions
## TODO remove P4
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
SetParameters.set_parameter(T::Type{<:AbstractFill}, ::Position{1}, P1) = T{P1}

# Set parameter 2
function SetParameters.set_parameter(
  T::Type{<:AbstractFill{P1}}, ::Position{2}, P2
) where {P1}
  return T{P2}
end

# Set parameter 3
function SetParameters.set_parameter(
  T::Type{<:AbstractFill{P1,P2}}, ::Position{3}, P3
) where {P1,P2}
  return T{P3}
end

## TODO define a specify_parameters function
## To quickly specify P1, P2, and P3 
## default parameters
function SetParameters.default_parameter(::Type{<:AbstractFill}, ::Position{1})
  return UnspecifiedTypes.UnallocatedZeros
end
SetParameters.default_parameter(::Type{<:AbstractFill}, ::Position{2}) = 0
SetParameters.default_parameter(::Type{<:AbstractFill}, ::Position{3}) = Tuple{}

SetParameters.nparameters(::Type{<:AbstractFill}) = Val(3)

# Set parameter 1
## Right now using AbstractArray
## TODO These are more difficult because T is technically defined so need some way to strip T of it {} types
# function set_parameter(
#   T::Type{<:AbstractFill{<:Any,<:Any,P3}}, ::Position{1}, P1
# ) where {P3}
#   return T{P1,<:Any,P3}
# end
# function set_parameter(
#   T::Type{<:AbstractFill{<:Any,P2,P3}}, ::Position{1}, P1
# ) where {P2,P3}
#   return T{P1,P2,P3}
# end
# function set_parameter(
#   T::Type{<:AbstractFill{<:Any,<:Any,P3}}, ::Position{2}, P2
# ) where {P3}
#   return T{<:Any,P2,P3}
# end
# function set_parameter(
#   T::Type{<:AbstractFill{P1,<:Any,P3}}, ::Position{2}, P2
# ) where {P1,P3}
#   return T
# end
# set_parameter(T::Type{<:AbstractFill}, ::Position{3}, P3) = T{<:Any,<:Any,P3}
# set_parameter(T::Type{<:AbstractFill{P1}}, ::Position{3}, P3) where {P1} = T{<:Any,P3}
# function set_parameter(T::Type{<:AbstractFill{<:Any,P2}}, ::Position{3}, P3) where {P2}
#   return T{<:Any,P2,P3}
# end

# Set parameter 2
## using AbstractArray
