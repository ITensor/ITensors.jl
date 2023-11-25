# `SetParameters.jl` overloads.
get_parameter(::Type{<:SubArray{P1}}, ::Position{1}) where {P1} = P1
get_parameter(::Type{<:SubArray{<:Any,P2}}, ::Position{2}) where {P2} = P2
get_parameter(::Type{<:SubArray{<:Any,<:Any,P3}}, ::Position{3}) where {P3} = P3
get_parameter(::Type{<:SubArray{<:Any,<:Any,<:Any,P4}}, ::Position{4}) where {P4} = P4
get_parameter(::Type{<:SubArray{<:Any,<:Any,<:Any,<:Any,P5}}, ::Position{5}) where {P5} = P5

# Set parameter 1
# TODO: Finish defining all setters, maybe use a macro to generate all combinations.
function set_parameter(
  ::Type{<:SubArray{<:Any,<:Any,<:Any,P4,P5}}, ::Position{1}, P1
) where {P4,P5}
  return SubArray{P1,<:Any,<:Any,P4,P5}
end
function set_parameter(
  ::Type{<:SubArray{<:Any,P2,<:Any,P4,P5}}, ::Position{1}, P1
) where {P2,P4,P5}
  return SubArray{P1,P2,P4,P5}
end
function set_parameter(
  ::Type{<:SubArray{<:Any,<:Any,P3,P4,P5}}, ::Position{1}, P1
) where {P3,P4,P5}
  return SubArray{P1,<:Any,P3,P4,P5}
end
function set_parameter(
  ::Type{<:SubArray{<:Any,P2,P3,P4,P5}}, ::Position{1}, P1
) where {P2,P3,P4,P5}
  return SubArray{P1,P2,P3,P4,P5}
end

## # Set parameter 2
## set_parameter(::Type{<:SubArray}, ::Position{2}, P2) = SubArray{<:Any,P2}
## set_parameter(::Type{<:SubArray{P1}}, ::Position{2}, P2) where {P1} = SubArray{P1,P2}
## function set_parameter(::Type{<:SubArray{<:Any,<:Any,P3}}, ::Position{2}, P2) where {P3}
##   return SubArray{<:Any,P2,P3}
## end
## function set_parameter(::Type{<:SubArray{P1,<:Any,P3}}, ::Position{2}, P2) where {P1,P3}
##   return SubArray{P1,P2,P3}
## end

# Set parameter 3
function set_parameter(
  ::Type{<:SubArray{<:Any,<:Any,<:Any,P4,P5}}, ::Position{3}, P3
) where {P4,P5}
  return SubArray{<:Any,<:Any,P3,P4,P5}
end
function set_parameter(
  ::Type{<:SubArray{P1,<:Any,<:Any,P4,P5}}, ::Position{3}, P3
) where {P1,P4,P5}
  return SubArray{P1,<:Any,P3,P4,P5}
end
function set_parameter(
  ::Type{<:SubArray{<:Any,P2,<:Any,P4,P5}}, ::Position{3}, P3
) where {P2,P4,P5}
  return SubArray{<:Any,P2,P3,P4,P5}
end
function set_parameter(
  ::Type{<:SubArray{P1,P2,<:Any,P4,P5}}, ::Position{3}, P3
) where {P1,P2,P4,P5}
  return SubArray{P1,P2,P3,P4,P5}
end

# TODO: Define `default_parameter`.

nparameters(::Type{<:SubArray}) = Val(5)
