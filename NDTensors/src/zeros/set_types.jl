import NDTensors.SetParameters: nparameters, get_parameter, set_parameter, default_parameter

# `SetParameters.jl` overloads.
get_parameter(::Type{<:Zeros{P1}}, ::Position{1}) where {P1} = P1
get_parameter(::Type{<:Zeros{<:Any,P2}}, ::Position{2}) where {P2} = P2
get_parameter(::Type{<:Zeros{<:Any,<:Any,P3}}, ::Position{3}) where {P3} = P3

# Set parameter 1
set_parameter(::Type{<:Zeros}, ::Position{1}, P1) = Zeros{P1}
set_parameter(::Type{<:Zeros{<:Any,P2}}, ::Position{1}, P1) where {P2} = Zeros{P1,P2}
function set_parameter(::Type{<:Zeros{<:Any,<:Any,P3}}, ::Position{1}, P1) where {P3}
  @assert eltype(P3) == P1
  return Zeros{P1,<:Any,P3}
end
function set_parameter(::Type{<:Zeros{<:Any,P2,P3}}, ::Position{1}, P1) where {P2,P3}
  @assert eltype(P3) == P1
  return Zeros{P1,P2,P3}
end

# Set parameter 2
set_parameter(::Type{<:Zeros}, ::Position{2}, P2) = Zeros{<:Any,P2}
set_parameter(::Type{<:Zeros{P1}}, ::Position{2}, P2) where {P1} = Zeros{P1,P2}
function set_parameter(::Type{<:Zeros{<:Any,<:Any,P3}}, ::Position{2}, P2) where {P3}
  P1 = eltype(P3)
  return Zeros{P1,P2,P3}
end
function set_parameter(::Type{<:Zeros{P1,<:Any,P3}}, ::Position{2}, P2) where {P1,P3}
  return Zeros{P1,P2,P3}
end

# Set parameter 3
set_parameter(::Type{<:Zeros}, ::Position{3}, P3) = Zeros{<:Any,<:Any,P3}
set_parameter(::Type{<:Zeros{P1}}, ::Position{3}, P3) where {P1} = Zeros{P1,<:Any,P3}
function set_parameter(::Type{<:Zeros{<:Any,P2}}, ::Position{3}, P3) where {P2}
  P1 = eltype(P3)
  return Zeros{P1,P2,P3}
end
set_parameter(::Type{<:Zeros{P1,P2}}, ::Position{3}, P3) where {P1,P2} = Zeros{P1,P2,P3}

default_parameter(::Type{<:Zeros}, ::Position{1}) = Float64
default_parameter(::Type{<:Zeros}, ::Position{2}) = 1
default_parameter(::Type{<:Zeros}, ::Position{3}) = Vector{Float64}

nparameters(::Type{<:Zeros}) = Val(3)
