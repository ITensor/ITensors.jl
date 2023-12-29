function getindex(tensor::MatrixOrArrayStorageTensor, I::Integer...)
  return storage(tensor)[I...]
end

function setindex!(tensor::MatrixOrArrayStorageTensor, v, I::Integer...)
  storage(tensor)[I...] = v
  return tensor
end
