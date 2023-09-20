import .SetParameters: set_parameter, nparameters, default_parameter

# `SetParameters.jl` overloads.
NDTensors.SetParameters.get_parameter(::Type{<:UnallocatedZeros{P1}}, ::Position{1}) where {P1} = P1
NDTensors.SetParameters.get_parameter(::Type{<:UnallocatedZeros{<:Any,P2}}, ::Position{2}) where {P2} = P2
NDTensors.SetParameters.get_parameter(::Type{<:UnallocatedZeros{<:Any,<:Any,P3}}, ::Position{3}) where {P3} = P3
function NDTensors.SetParameters.get_parameter(
  ::Type{<:UnallocatedZeros{<:Any,<:Any,<:Any,P4}}, ::Position{4}
) where {P4}
  return P4
end

# Set parameter 1
function set_parameter(
  ::Type{<:UnallocatedZeros{<:Any,P2,P3,P4}}, ::Position{1}, P1
) where {P2,P3,P4}
  alloc = similartype(P4, P1)
  return UnallocatedZeros{P1,P2,P3,alloc}
end

# Set parameter 2
function set_parameter(
  ::Type{<:UnallocatedZeros{P1,<:Any,P3,P4}}, ::Position{2}, P2
) where {P1,P3,P4}
  return UnallocatedZeros{P1,P2,P3,P4}
end

# Set parameter 3
function set_parameter(
  ::Type{<:UnallocatedZeros{P1,P2,<:Any,P4}}, ::Position{3}, P3
) where {P1,P2,P4}
  return UnallocatedZeros{P1,P2,P3,P4}
end

# Set parameter 4
function set_parameter(
  ::Type{<:UnallocatedZeros{P1,P2,P3,<:Any}}, ::Position{4}, P4
) where {P1,P2,P3}
@show P4
  return UnallocatedZeros{P1,P2,P3,P4}
end

default_parameter(::Type{<:UnallocatedZeros}, ::Position{1}) = Float64
default_parameter(::Type{<:UnallocatedZeros}, ::Position{2}) = 1
default_parameter(::Type{<:UnallocatedZeros}, ::Position{3}) = Tuple{Base.OneTo{Int64}}
default_parameter(::Type{<:UnallocatedZeros}, ::Position{4}) = Vector{default_parameter(UnallocatedZeros, Position(1))}

nparameters(::Type{<:UnallocatedZeros}) = Val(4)
