function set_eltype(arraytype::Type{<:Tensor}, eltype::Type)
  return set_storagetype(arraytype, set_eltype(storagetype(arraytype), eltype))
end

function set_ndims(arraytype::Type{<:Tensor}, ndims)
  # TODO: Implement something like:
  # ```julia
  # return set_storagetype(arraytype, set_ndims(storagetype(arraytype), ndims))
  # ```
  # However, we will also need to define `set_ndims(indstype(arraytype), ndims)`
  # and use `set_indstype(arraytype, set_ndims(indstype(arraytype), ndims))`.
  return error(
    "Setting the number dimensions of the array type `$arraytype` (to `$ndims`) is not currently defined.",
  )
end
