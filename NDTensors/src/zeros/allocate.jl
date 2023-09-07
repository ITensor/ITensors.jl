function allocate(T::Tensor, elt::Type=default_eltype())
  if !is_unallocated_zeros(T)
    return T
  end
  ## allocate the tensor if is_unallocated_zeros
  store = allocate(storage(T), elt)
  return Tensor(NDTensors.AllowAlias(), store, inds(T))
end

function allocate(storage::TensorStorage, elt::Type=default_eltype())
  if !is_unallocated_zeros(storage)
    return storage
  end
  alloc = allocate(data(storage), elt)

  #d = adapt(storage, typeof(alloc))(alloc)
  return set_datatype(typeof(storage), typeof(alloc))(alloc)
end

allocate(d::AbstractArray, elt::Type=default_eltype()) = d

function allocate(z::UnallocatedZeros, elt::Type=default_eltype())
  alloc = specify_eltype(alloctype(z), elt)(undef, dims(z))
  return fill!(alloc, elt(0))
end
