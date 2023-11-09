## TODO I have just started working on this, it needs more work still.
for Typ in (:UnallocatedFill, :UnallocatedZeros)
  @eval begin
    # `SetParameters.jl` overloads.
    get_parameter(::Type{<:$Typ{P1}}, ::Position{1}) where {P1} = P1
    get_parameter(::Type{<:$Typ{<:Any,P2}}, ::Position{2}) where {P2} = P2
    get_parameter(::Type{<:$Typ{<:Any,<:Any,P3}}, ::Position{3}) where {P3} = P3
    get_parameter(
      ::Type{<:$Typ{<:Any,<:Any,<:Any,P4}}, ::Position{4}
    ) where {P4} = P4

    ## Setting paramaters
    # Set parameter 1
    set_parameter(::Type{<:$Typ}, ::Position{1}, P1) = $Typ{P1}
    set_parameter(::Type{<:$Typ{<:Any,P2}}, ::Position{1}, P1) where {P2} = $Typ{P1,P2}
    function set_parameter(::Type{<:$Typ{<:Any,<:Any,P3}}, ::Position{1}, P1) where {P3}
      return $Typ{P1,<:Any,P3}
    end
    function set_parameter(::Type{<:$Typ{<:Any,P2,P3}}, ::Position{1}, P1) where {P2,P3}
      return $Typ{P1,P2,P3}
    end
    function set_parameter(::Type{<:$Typ{<:Any,P2,P3,P4}}, ::Position{1}, P1) where {P2,P3,P4}
      return $Typ{P1,P2,P3,P4}
    end
    function set_parameter(::Type{<:$Typ{<:Any,<:Any,P3,P4}}, ::Position{1}, P1) where {P3,P4}
      return $Typ{P1,<:Any,P3,P4}
    end
    function set_parameter(::Type{<:$Typ{<:Any,P2,<:Any,P4}}, ::Position{1}, P1) where {P2,P4}
      return $Typ{P1,P2,<:Any,P4}
    end
    function set_parameter(::Type{<:$Typ{<:Any,<:Any,<:Any,P4}}, ::Position{1}, P1) where {P4}
      return $Typ{P1,<:Any,<:Any,P4}
    end

    # Set parameter 2
    set_parameter(::Type{<:$Typ}, ::Position{2}, P2) = $Typ{<:Any,P2}
    set_parameter(::Type{<:$Typ{P1}}, ::Position{2}, P2) where {P1} = $Typ{P1,P2}
    function set_parameter(::Type{<:$Typ{<:Any,<:Any,P3}}, ::Position{2}, P2) where {P3}
      return $Typ{<:Any,P2,P3}
    end
    function set_parameter(::Type{<:$Typ{P1,<:Any,P3}}, ::Position{2}, P2) where {P1,P3}
      return $Typ{P1,P2,P3}
    end

    # Set parameter 3
    set_parameter(::Type{<:$Typ}, ::Position{3}, P3) = $Typ{<:Any,<:Any,P3}
    set_parameter(::Type{<:$Typ{P1}}, ::Position{3}, P3) where {P1} = $Typ{P1,<:Any,P3}
    function set_parameter(::Type{<:$Typ{<:Any,P2}}, ::Position{3}, P3) where {P2}
      return $Typ{<:Any,P2,P3}
    end
    set_parameter(::Type{<:$Typ{P1,P2}}, ::Position{3}, P3) where {P1,P2} = $Typ{P1,P2,P3}

    # Set paramter 4
    set_parameter(::Type{<:$Typ}, ::Position{4}, P4) = $Typ{<:Any,<:Any,<:Any,P4}
    set_parameter(::Type{<:$Typ{P1}}, ::Position{4}, P4) where {P1} = $Typ{P1,<:Any,<:Any,P4}
    set_parameter(::Type{<:$Typ{P1,P2}}, ::Position{4}, P4) where{P1,P2} = $Typ{P1,P2,<:Any,P4}
    set_parameter(::Type{<:$Typ{P1,P2,P3}}, ::Position{4}, P4) where{P1,P2,P3} = $Typ{P1,P2,P3,P4}

## default parameters
    default_parameter(::Type{<:$Typ}, ::Position{1}) =
      UnspecifiedTypes.UnallocatedZeros
    default_parameter(::Type{<:$Typ}, ::Position{2}) = 0
    default_parameter(::Type{<:$Typ}, ::Position{3}) = Tuple{}
    default_parameter(::Type{<:$Typ}, ::Position{4}) =
      UnspecifiedTypes.UnspecifiedArray

    nparameters(::Type{<:$Typ}) = Val(4)
  end
end
