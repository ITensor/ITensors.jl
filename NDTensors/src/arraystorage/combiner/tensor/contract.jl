# TODO: These should be generic `Tensor` definitions.
function contract(
  t1::Tensor{<:Any,<:Any,<:CombinerArray}, labels1, t2::MatrixOrArrayStorageTensor, labels2
)
  a_dest, labels_dest = contract(storage(t1), labels1, storage(t2), labels2)
  inds_dest = contract_inds(inds(t1), labels1, inds(t2), labels2, labels_dest)
  return tensor(a_dest, inds_dest)
end

function contract(
  t1::MatrixOrArrayStorageTensor, labels1, t2::Tensor{<:Any,<:Any,<:CombinerArray}, labels2
)
  a_dest, labels_dest = contract(storage(t1), labels1, storage(t2), labels2)
  inds_dest = contract_inds(inds(t1), labels1, inds(t2), labels2, labels_dest)
  return tensor(a_dest, inds_dest)
end
