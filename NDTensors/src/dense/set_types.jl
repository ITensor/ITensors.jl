using .SetParameters:
  SetParameters, Position, get_parameters, specify_parameters, unspecify_parameters
using .TypeParameterAccessors: TypeParameterAccessors, parenttype

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
SetParameters.default_parameter(::Type{<:Dense}, ::Position{1}) = Float64
SetParameters.default_parameter(::Type{<:Dense}, ::Position{2}) = Vector

SetParameters.set_parameter(::Type{<:Dense}, ::Position{1}, P1) = Dense{P1}
function SetParameters.set_parameter(
  ::Type{<:Dense{<:Any,P2}}, ::Position{1}, P1
) where {P2}
  return Dense{P1,P2}
end

SetParameters.set_parameter(::Type{<:Dense}, ::Position{2}, P2) = Dense{<:Any,P2}
function SetParameters.set_parameter(::Type{<:Dense{P1}}, ::Position{2}, P2) where {P1}
  return Dense{P1,P2}
end
TypeParameterAccessors.position(::Type{<:Dense}, ::typeof(parenttype)) = TypeParameterAccessors.Position(2)
TypeParameterAccessors.default_type_parameters(::Type{<:Dense}) = (Float64, Vector)
