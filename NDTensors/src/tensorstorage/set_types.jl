using .TypeParameterAccessor: TypeParameterAccessor
function TypeParameterAccessor.set_ndims(arraytype::Type{<:TensorStorage}, ndims::Int)
  # TODO: Change to this once `TensorStorage` types support wrapping
  # non-AbstractVector types.
  # return set_datatype(arraytype, set_ndims(datatype(arraytype), ndims))
  return arraytype
end
