using .TypeParameterAccessors: TypeParameterAccessors, Position

function set_datatype(storagetype::Type{<:Dense}, datatype::Type{<:AbstractVector})
  return Dense{eltype(datatype),datatype}
end

function set_datatype(storagetype::Type{<:Dense}, datatype::Type{<:AbstractArray})
  return error(
    "Setting the `datatype` of the storage type `$storagetype` to a $(ndims(datatype))-dimsional array of type `$datatype` is not currently supported, use an `AbstractVector` instead.",
  )
end

TypeParameterAccessors.position(::Type{<:Dense}, ::typeof(TypeParameterAccessors.parenttype)) = Position(2)
TypeParameterAccessors.default_parameter(::Type{<:Dense}, ::typeof(eltype)) = Float64
TypeParameterAccessors.default_parameter(::Type{<:Dense}, ::typeof(TypeParameterAccessors.parenttype)) = Vector
