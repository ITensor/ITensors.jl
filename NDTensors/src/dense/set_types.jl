using .TypeParameterAccessor:
TypeParameterAccessor, Position, parameters, specify_parameters, unspecify_parameters

function set_datatype(storagetype::Type{<:Dense}, datatype::Type{<:AbstractVector})
  return Dense{eltype(datatype),datatype}
end

function set_datatype(storagetype::Type{<:Dense}, datatype::Type{<:AbstractArray})
  return error(
    "Setting the `datatype` of the storage type `$storagetype` to a $(ndims(datatype))-dimsional array of type `$datatype` is not currently supported, use an `AbstractVector` instead.",
  )
end

# TypeParameterAccessor.unspecify_parameters(::Type{<:Dense}) = Dense

TypeParameterAccessor.parenttype_position(::Type{<:Dense}) = Position(2)
TypeParameterAccessor.default_parameter(::Type{<:Dense}, ::Position{1}) = Float64
TypeParameterAccessor.default_parameter(::Type{<:Dense}, ::Position{2}) = Vector
