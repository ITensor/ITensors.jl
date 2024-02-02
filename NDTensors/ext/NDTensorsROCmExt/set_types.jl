# `SetParameters.jl` overloads.
get_parameter(::Type{<:ROCArray{P1}}, ::Position{1}) where {P1} = P1
get_parameter(::Type{<:ROCArray{<:Any,P2}}, ::Position{2}) where {P2} = P2
get_parameter(::Type{<:ROCArray{<:Any,<:Any,P3}}, ::Position{3}) where {P3} = P3

# Set parameter 1
set_parameter(::Type{<:ROCArray}, ::Position{1}, P1) = ROCArray{P1}
set_parameter(::Type{<:ROCArray{<:Any,P2}}, ::Position{1}, P1) where {P2} = ROCArray{P1,P2}
function set_parameter(::Type{<:ROCArray{<:Any,<:Any,P3}}, ::Position{1}, P1) where {P3}
  return ROCArray{P1,<:Any,P3}
end
function set_parameter(::Type{<:ROCArray{<:Any,P2,P3}}, ::Position{1}, P1) where {P2,P3}
  return ROCArray{P1,P2,P3}
end

# Set parameter 2
set_parameter(::Type{<:ROCArray}, ::Position{2}, P2) = ROCArray{<:Any,P2}
set_parameter(::Type{<:ROCArray{P1}}, ::Position{2}, P2) where {P1} = ROCArray{P1,P2}
function set_parameter(::Type{<:ROCArray{<:Any,<:Any,P3}}, ::Position{2}, P2) where {P3}
  return ROCArray{<:Any,P2,P3}
end
function set_parameter(::Type{<:ROCArray{P1,<:Any,P3}}, ::Position{2}, P2) where {P1,P3}
  return ROCArray{P1,P2,P3}
end

# Set parameter 3
set_parameter(::Type{<:ROCArray}, ::Position{3}, P3) = ROCArray{<:Any,<:Any,P3}
set_parameter(::Type{<:ROCArray{P1}}, ::Position{3}, P3) where {P1} = ROCArray{P1,<:Any,P3}
function set_parameter(::Type{<:ROCArray{<:Any,P2}}, ::Position{3}, P3) where {P2}
  return ROCArray{<:Any,P2,P3}
end
function set_parameter(::Type{<:ROCArray{P1,P2}}, ::Position{3}, P3) where {P1,P2}
  return ROCArray{P1,P2,P3}
end

default_parameter(::Type{<:ROCArray}, ::Position{1}) = Float64
default_parameter(::Type{<:ROCArray}, ::Position{2}) = 1
default_parameter(::Type{<:ROCArray}, ::Position{3}) = Mem.HIPBuffer

nparameters(::Type{<:ROCArray}) = Val(3)
