using .TypeParameterAccessors: TypeParameterAccessors, Position

function set_datatype(storagetype::Type{<:Dense}, datatype::Type{<:AbstractVector})
  return Dense{eltype(datatype),datatype}
end

function set_datatype(storagetype::Type{<:Dense}, datatype::Type{<:AbstractArray})
  return error(
    "Setting the `datatype` of the storage type `$storagetype` to a $(ndims(datatype))-dimsional array of type `$datatype` is not currently supported, use an `AbstractVector` instead.",
  )
end

function TypeParameterAccessors.position(
  ::Type{<:Dense}, ::typeof(TypeParameterAccessors.parenttype)
)
  return Position(2)
end
TypeParameterAccessors.default_parameter(::Type{<:Dense}, ::typeof(eltype)) = Float64
function TypeParameterAccessors.default_parameter(
  ::Type{<:Dense}, ::typeof(TypeParameterAccessors.parenttype)
)
  return Vector
end
