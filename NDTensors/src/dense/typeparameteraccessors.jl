using .TypeParameterAccessors:
  TypeParameterAccessors,
  default_type_parameter,
  default_type_parameters,
  parenttype,
  position,
  unwrap_array_type,
  set_parenttype

## Dense
datatype(storetype::Type{<:Dense}) = parenttype(storetype)

function set_datatype(storagetype::Type{<:Dense}, datatype::Type{<:AbstractVector})
  return set_parenttype(storagetype, datatype)
end

function set_datatype(storagetype::Type{<:Dense}, datatype::Type{<:AbstractArray})
  return error(
    "Setting the `datatype` of the storage type `$storagetype` to a $(ndims(datatype))-dimsional array of type `$datatype` is not currently supported, use an `AbstractVector` instead.",
  )
end

function Base.real(T::Type{<:Dense})
  return set_datatype(T, similartype(datatype(T), real(eltype(T))))
end

function complex(T::Type{<:Dense})
  return set_datatype(T, similartype(datatype(T), complex(eltype(T))))
end

function TypeParameterAccessors.position(::Type{<:Dense}, ::typeof(eltype))
  return TypeParameterAccessors.Position(1)
end
function TypeParameterAccessors.position(::Type{<:Dense}, ::typeof(parenttype))
  return TypeParameterAccessors.Position(2)
end

function TypeParameterAccessors.default_type_parameters(::Type{<:Dense})
  return (
    default_type_parameter(Vector, eltype), Vector{default_type_parameter(Vector, eltype)}
  )
end

function TypeParameterAccessors.position(::Type{<:DenseTensor}, ::typeof(Base.ndims))
  return TypeParameterAccessors.Position(2)
end

function TypeParameterAccessors.set_ndims(type::Type{<:DenseTensor}, N)
  return set_type_parameter(type, Base.ndims, N)
end
