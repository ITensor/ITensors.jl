# `SetParameters.jl` overloads.
get_parameter(::Type{<:CuArray{P1}}, ::Position{1}) where {P1} = P1
get_parameter(::Type{<:CuArray{<:Any,P2}}, ::Position{2}) where {P2} = P2
get_parameter(::Type{<:CuArray{<:Any,<:Any,P3}}, ::Position{3}) where {P3} = P3

# Set parameter 1
set_parameter(::Type{<:CuArray}, ::Position{1}, P1) = CuArray{P1}
set_parameter(::Type{<:CuArray{<:Any,P2}}, ::Position{1}, P1) where {P2} = CuArray{P1,P2}
function set_parameter(::Type{<:CuArray{<:Any,<:Any,P3}}, ::Position{1}, P1) where {P3}
  return CuArray{P1,<:Any,P3}
end
function set_parameter(::Type{<:CuArray{<:Any,P2,P3}}, ::Position{1}, P1) where {P2,P3}
  return CuArray{P1,P2,P3}
end

# Set parameter 2
set_parameter(::Type{<:CuArray}, ::Position{2}, P2) = CuArray{<:Any,P2}
set_parameter(::Type{<:CuArray{P1}}, ::Position{2}, P2) where {P1} = CuArray{P1,P2}
function set_parameter(::Type{<:CuArray{<:Any,<:Any,P3}}, ::Position{2}, P2) where {P3}
  return CuArray{<:Any,P2,P3}
end
function set_parameter(::Type{<:CuArray{P1,<:Any,P3}}, ::Position{2}, P2) where {P1,P3}
  return CuArray{P1,P2,P3}
end

# Set parameter 3
set_parameter(::Type{<:CuArray}, ::Position{3}, P3) = CuArray{<:Any,<:Any,P3}
set_parameter(::Type{<:CuArray{P1}}, ::Position{3}, P3) where {P1} = CuArray{P1,<:Any,P3}
function set_parameter(::Type{<:CuArray{<:Any,P2}}, ::Position{3}, P3) where {P2}
  return CuArray{<:Any,P2,P3}
end
set_parameter(::Type{<:CuArray{P1,P2}}, ::Position{3}, P3) where {P1,P2} = CuArray{P1,P2,P3}

default_parameter(::Type{<:CuArray}, ::Position{1}) = Float64
default_parameter(::Type{<:CuArray}, ::Position{2}) = 1
default_parameter(::Type{<:CuArray}, ::Position{3}) = Mem.DeviceBuffer

nparameters(::Type{<:CuArray}) = Val(3)
