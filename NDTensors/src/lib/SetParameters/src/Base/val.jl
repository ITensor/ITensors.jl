# `SetParameters.jl` overloads.
get_parameter(::Type{<:Val{P1}}, ::Position{1}) where {P1} = P1

set_parameter(::Type{<:Val}, ::Position{1}, P1) = Val{P1}

nparameters(::Type{<:Val}) = Val(1)
