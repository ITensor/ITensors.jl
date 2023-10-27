function permutedims!(
  output_tensor::MatrixOrArrayStorageTensor,
  tensor::MatrixOrArrayStorageTensor,
  perm,
  f::Function,
)
  permutedims!(storage(output_tensor), storage(tensor), perm, f)
  return output_tensor
end
