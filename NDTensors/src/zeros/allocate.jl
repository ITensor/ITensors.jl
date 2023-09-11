alloctype(a::AbstractArray) = a
## TODO remove this when change Diag
alloctype(a::Number) = a
function allocate(T::Tensor)
  return adapt(alloctype(data(T)), T)
end

function allocate(T::Type{<:Tensor}, inds::Tuple)
  store = set_datatype(storagetype(T), alloctype(datatype(T)))(inds)
  tensor(store, inds)
end

function allocate(storage::TensorStorage)
  return adapt(alloctype(data(storage)), storage)
end

function allocate(storage::Type{<:TensorStorage}, inds::Tuple)
  return set_datatype(storage, alloctype(datatype(storage)))(inds)
end

allocate(d::AbstractArray) = d

function allocate(z::UnallocatedZeros)
  return alloctype(z)(data(z))
end
