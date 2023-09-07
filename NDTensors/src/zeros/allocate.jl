function allocate(T::Tensor)
  return adapt(alloctype(data(T)), T)
end

function allocate(storage::TensorStorage)
  return adapt(alloctype(data(storage)), storage)
end

allocate(d::AbstractArray) = d

function allocate(z::UnallocatedZeros)
  return alloctype(z)(data(z))
end
