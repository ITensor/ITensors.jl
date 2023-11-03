# TODO: Make a `CombinerArray`.
function contract!!(
  tensor_dest::ArrayStorageTensor,
  tensor_dest_labels::Any,
  tensor_src::Tensor{T,N,<:BlockSparseArray{T,N}},
  tensor_src_labels,
  tensor_combiner::CombinerTensor,
  tensor_combiner_labels,
) where {T,N}
  array_dest = contract!!(storage(tensor_dest), tensor_dest_labels, storage(tensor_src), tensor_src_labels, tensor_combiner, tensor_combiner_labels)
  # TODO: Check if this is correct.
  # Might be incorrect for uncombining!
  inds_dest = contract_inds(
    inds(tensor_combiner),
    tensor_combiner_labels,
    inds(tensor_src),
    tensor_src_labels,
    tensor_dest_labels,
  )
  return tensor(array_dest, inds_dest)
end

function contract!!(
  tensor_dest::ArrayStorageTensor,
  tensor_dest_labels::Any,
  combiner_tensor::CombinerTensor,
  combiner_tensor_labels,
  tensor::Tensor{T,N,<:BlockSparseArray{T,N}},
  tensor_labels,
) where {T,N}
  return contract!!(tensor_dest, tensor_dest_labels, tensor, tensor_labels, combiner_tensor, combiner_tensor_labels)
end
