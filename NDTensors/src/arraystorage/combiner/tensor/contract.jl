function contract(
  t_comb::Tensor{<:Any,<:Any,<:CombinerArray},
  labels_comb,
  t_src::MatrixOrArrayStorageTensor,
  labels_src,
)
  a_dest, labels_dest = contract(storage(t_comb), labels_comb, storage(t_src), labels_src)
  inds_dest = contract_inds(inds(t_comb), labels_comb, inds(t_src), labels_src, labels_dest)
  return tensor(a_dest, inds_dest)
end

function contract(
  t_src::MatrixOrArrayStorageTensor,
  labels_src,
  t_comb::Tensor{<:Any,<:Any,<:CombinerArray},
  labels_comb,
)
  return contract(t_comb, labels_comb, t_src, labels_src)
end
