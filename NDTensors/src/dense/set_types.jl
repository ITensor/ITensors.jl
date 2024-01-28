using .SetParameters:
  SetParameters, Position, get_parameters, specify_parameters, unspecify_parameters

function set_datatype(storagetype::Type{<:Dense}, datatype::Type{<:AbstractVector})
  return Dense{eltype(datatype),datatype}
end

function set_datatype(storagetype::Type{<:Dense}, datatype::Type{<:AbstractArray})
  return error(
    "Setting the `datatype` of the storage type `$storagetype` to a $(ndims(datatype))-dimsional array of type `$datatype` is not currently supported, use an `AbstractVector` instead.",
  )
end

SetParameters.unspecify_parameters(::Type{<:Dense}) = Dense

SetParameters.parenttype_position(::Type{<:Dense}) = Position(2)
SetParameters.nparameters(::Type{<:Dense}) = Val(2)
SetParameters.get_parameter(::Type{<:Dense{P1}}, ::Position{1}) where {P1} = P1
SetParameters.get_parameter(::Type{<:Dense{<:Any,P2}}, ::Position{2}) where {P2} = P2

function SetParameters.set_parameter(t::Type{<:Dense}, ::Position{1}, P1)
  return specify_parameters(unspecify_parameters(t){P1}, get_parameters(t))
end

function SetParameters.set_parameter(t::Type{<:Dense}, ::Position{2}, P2)
  P1 = get_parameter(t, Position(1))
  return specify_parameters(unspecify_parameters(t){P1,P2}, get_parameters(t))
end
