using .TypeParameterAccessors: TypeParameterAccessors, Position, type_parameter
datatype(type::Type{<:TensorStorage}) = type_parameter(type, parenttype)
datatype(S::TensorStorage) = datatype(typeof(S))

TypeParameterAccessors.position(::Type{<:TensorStorage}, ::typeof(eltype)) = Position(1)

function TypeParameterAccessors.set_ndims(arraytype::Type{<:TensorStorage}, ndims::Int)
  # TODO: Change to this once `TensorStorage` types support wrapping
  # non-AbstractVector types.
  # return (arraytype, set_ndims(datatype(arraytype), ndims))
  return arraytype
end
