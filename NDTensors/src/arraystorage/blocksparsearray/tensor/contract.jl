# TODO: This should be a generic `Tensor` definition.
function contract(
  t1::Tensor{<:Any,<:Any,<:BlockSparseArray}, labels1, t2::Tensor{<:Any,<:Any,<:BlockSparseArray}, labels2
)
  a_dest, labels_dest = contract(storage(t1), labels1, storage(t2), labels2)
  inds_dest = contract_inds(inds(t1), labels1, inds(t2), labels2, labels_dest)
  return tensor(a_dest, inds_dest)
end

# TODO: This should be a generic `Tensor` definition.
function contract(
  t1::Tensor{<:Any,<:Any,<:BlockSparseArray}, labels1, t2::MatrixOrArrayStorageTensor, labels2
)
  a_dest, labels_dest = contract(storage(t1), labels1, storage(t2), labels2)
  inds_dest = contract_inds(inds(t1), labels1, inds(t2), labels2, labels_dest)
  return tensor(a_dest, inds_dest)
end

# TODO: This should be a generic `Tensor` definition.
function contract(
  t1::MatrixOrArrayStorageTensor, labels1, t2::Tensor{<:Any,<:Any,<:BlockSparseArray}, labels2
)
  a_dest, labels_dest = contract(storage(t1), labels1, storage(t2), labels2)
  inds_dest = contract_inds(inds(t1), labels1, inds(t2), labels2, labels_dest)
  return tensor(a_dest, inds_dest)
end

# TODO: This should be a generic `Tensor` definition.
function contract(
  t1::Tensor{<:Any,<:Any,<:CombinerArray}, labels1, t2::Tensor{T,N,<:BlockSparseArray{T,N}}, labels2
) where {T,N}
  a_dest, labels_dest = contract(storage(t1), labels1, storage(t2), labels2)
  inds_dest = contract_inds(inds(t1), labels1, inds(t2), labels2, labels_dest)
  return tensor(a_dest, inds_dest)
end

# TODO: This should be a generic `Tensor` definition.
function contract(
  t1::Tensor{T,N,<:BlockSparseArray{T,N}}, labels1, t2::Tensor{<:Any,<:Any,<:CombinerArray}, labels2
) where {T,N}
  a_dest, labels_dest = contract(storage(t1), labels1, storage(t2), labels2)
  inds_dest = contract_inds(inds(t1), labels1, inds(t2), labels2, labels_dest)
  return tensor(a_dest, inds_dest)
end
