function set_eltype(arraytype::Type{<:TensorStorage}, eltype::Type)
  return set_datatype(arraytype, set_eltype(datatype(arraytype), eltype))
end

function set_ndims(arraytype::Type{<:TensorStorage}, ndims)
  # TODO: Change to this once `TensorStorage` types support wrapping
  # non-AbstractVector types.
  # return set_datatype(arraytype, set_ndims(datatype(arraytype), ndims))
  return arraytype
end
