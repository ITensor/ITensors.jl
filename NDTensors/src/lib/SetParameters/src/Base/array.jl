# `SetParameters.jl` overloads.
get_parameter(::Type{<:Array{P1}}, ::Position{1}) where {P1} = P1
get_parameter(::Type{<:Array{<:Any,P2}}, ::Position{2}) where {P2} = P2

set_parameter(::Type{<:Array}, ::Position{1}, P1) = Array{P1}
set_parameter(::Type{<:Array{<:Any,P2}}, ::Position{1}, P1) where {P2} = Array{P1,P2}
set_parameter(::Type{<:Array}, ::Position{2}, P2) = Array{<:Any,P2}
set_parameter(::Type{<:Array{P1}}, ::Position{2}, P2) where {P1} = Array{P1,P2}

default_parameter(::Type{<:Array}, ::Position{1}) = Float64
default_parameter(::Type{<:Array}, ::Position{2}) = 1

nparameters(::Type{<:Array}) = Val(2)
