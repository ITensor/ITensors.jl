function set_eltype(arraytype::Type{<:TensorStorage}, eltype::Type)
  return set_datatype(arraytype, set_eltype(datatype(arraytype), eltype))
end

function set_ndims(arraytype::Type{<:TensorStorage}, ndims)
  return set_datatype(arraytype, set_ndims(datatype(arraytype), ndims))
end
