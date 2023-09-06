function allocate(T::Tensor)
  if !is_unallocated_zeros(T)
    return T
  end
  return tensor(
    set_datatype(typeof(NDTensors.storage(T)), alloctype(data(T)))(allocate(data(T))),
    inds(T),
  )
  #@show convert(type, out_data)
  #return type(allocate(data(T)), inds(T))
end

function allocate(T::Tensor, elt::Type)
  if !is_unallocated_zeros(T)
    return T
  end
  ## allocate the tensor if is_unallocated_zeros
  ElT = promote_type(eltype(data(T)), elt)
  d = similartype(alloctype(data(T)), ElT)(undef, dim(to_shape(typeof(data(T)), inds(T))))
  fill!(d, 0)
  return Tensor(d, inds(T))
end

allocate(d::AbstractArray) = d

function allocate(z::UnallocatedZeros)
  return alloctype(z)(undef, dims(z))
end
