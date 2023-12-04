## TODO make unit tests for all of these functions
## TODO remove P4
# `SetParameters.jl` overloads.
get_parameter(::Type{<:AbstractFill{P1}}, ::Position{1}) where {P1} = P1
get_parameter(::Type{<:AbstractFill{<:Any,P2}}, ::Position{2}) where {P2} = P2
get_parameter(::Type{<:AbstractFill{<:Any,<:Any,P3}}, ::Position{3}) where {P3} = P3

## Setting paramaters
# Set parameter 1
set_parameter(T::Type{<:AbstractFill}, ::Position{1}, P1) = T{P1}
set_parameter(T::Type{<:AbstractFill{<:Any,P2}}, ::Position{1}, P1) where {P2} = T{P1,P2}
function set_parameter(
  T::Type{<:AbstractFill{<:Any,<:Any,P3}}, ::Position{1}, P1
) where {P3}
  return T{P1,<:Any,P3}
end
function set_parameter(
  T::Type{<:AbstractFill{<:Any,P2,P3}}, ::Position{1}, P1
) where {P2,P3}
  return T{P1,P2,P3}
end

# Set parameter 2
set_parameter(T::Type{<:AbstractFill}, ::Position{2}, P2) = T{<:Any,P2}
set_parameter(T::Type{<:AbstractFill{P1}}, ::Position{2}, P2) where {P1} = T{P1,P2}
function set_parameter(
  T::Type{<:AbstractFill{<:Any,<:Any,P3}}, ::Position{2}, P2
) where {P3}
  return T{<:Any,P2,P3}
end
function set_parameter(
  T::Type{<:AbstractFill{P1,<:Any,P3}}, ::Position{2}, P2
) where {P1,P3}
  return T{P1,P2,P3}
end

# Set parameter 3
set_parameter(T::Type{<:AbstractFill}, ::Position{3}, P3) = T{<:Any,<:Any,P3}
set_parameter(T::Type{<:AbstractFill{P1}}, ::Position{3}, P3) where {P1} = T{P1,<:Any,P3}
function set_parameter(T::Type{<:AbstractFill{<:Any,P2}}, ::Position{3}, P3) where {P2}
  return T{<:Any,P2,P3}
end
set_parameter(T::Type{<:AbstractFill{P1,P2}}, ::Position{3}, P3) where {P1,P2} = T{P1,P2,P3}

## default parameters
default_parameter(::Type{<:AbstractFill}, ::Position{1}) = UnspecifiedTypes.UnallocatedZeros
default_parameter(::Type{<:AbstractFill}, ::Position{2}) = 0
default_parameter(::Type{<:AbstractFill}, ::Position{3}) = Tuple{}

nparameters(::Type{<:AbstractFill}) = Val(3)
