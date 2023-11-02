## TODO I have just started working on this, it needs more work still.
for Typ in (:UnallocatedFill, :UnallocatedZeros)
  # `SetParameters.jl` overloads.
  get_parameter(::Type{<:$Typ{P1}}, ::Position{1}) where {P1} = P1
  get_parameter(::Type{<:$Typ{<:Any,P2}}, ::Position{2}) where {P2} = P2
  get_parameter(::Type{<:$Typ{<:Any,<:Any,P3}}, ::Position{3}) where {P3} = P3
  get_parameter(::Type{<:$Typ{<:Any,<:Any,<:Any,P4}}, ::Position{4}) where {P4} = P4

  default_parameter(::Type{<:$Typ}, ::Position{1}) = UnspecifiedTypes.UnallocatedZeros
  default_parameter(::Type{<:$Typ}, ::Position{2}) = 0
  default_parameter(::Type{<:$Typ}, ::Position{3}) = Tuple{}
  default_parameter(::Type{<:$Typ}, ::Position{4}) = UnspecifiedTypes.UnspecifiedArray

  nparameters(::Type{<:$Typ}) = Val(4)
end
