function permutedims!(
  tensor_dest::MatrixOrArrayStorageTensor,
  tensor_src::MatrixOrArrayStorageTensor,
  perm,
  f::Function,
)
  permutedims!(storage(tensor_dest), storage(tensor_src), perm, f)
  return tensor_dest
end

function permutedims(t::MatrixOrArrayStorageTensor, perm)
  a_perm = permutedims(storage(t), perm)
  return tensor(a_perm, permute(inds(t), perm))
end
