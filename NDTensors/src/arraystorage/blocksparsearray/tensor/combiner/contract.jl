# TODO: Make a `CombinerArray`.
function contract(
  tensor_src::Tensor{T,N,<:BlockSparseArray{T,N}},
  tensor_src_labels,
  tensor_combiner::Tensor{Any,M,<:CombinerArray{M}},
  tensor_combiner_labels,
) where {T,N,M}
  array_dest = contract(
    storage(tensor_src), tensor_src_labels, storage(tensor_combiner), tensor_combiner_labels
  )
  tensor_dest_inds =
    if is_combining(
      storage(tensor_src),
      tensor_src_labels,
      storage(tensor_combiner),
      tensor_combiner_labels,
    )
      contract_combine_inds(
        tensor_src, tensor_src_labels, tensor_combiner, tensor_combiner_labels
      )
    else
      contract_uncombine_inds(
        tensor_src, tensor_src_labels, tensor_combiner, tensor_combiner_labels
      )
    end
  return tensor(array_dest, tensor_dest_inds)
end

function contract(
  tensor_combiner::Tensor{Any,M,<:CombinerArray{M}},
  tensor_combiner_labels,
  tensor_src::Tensor{T,N,<:BlockSparseArray{T,N}},
  tensor_src_labels,
) where {T,N,M}
  return contract(tensor_src, tensor_src_labels, tensor_combiner, tensor_combiner_labels)
end

# TODO: Move to `contract_combine.jl`.
function contract_combine_inds(
  tensor_src::Tensor{<:Any,<:Any,<:BlockSparseArray},
  tensor_src_labels,
  tensor_combiner::Tensor{<:Any,<:Any,<:CombinerArray},
  tensor_combiner_labels,
)
  # Get the label marking the combined index
  # By convention the combined index is the first one
  # TODO: Consider storing the location of the combined
  # index in preperation for multiple combined indices
  # TODO: Use `combinedind_label(...)`, `uncombinedind_labels(...)`, etc.
  cpos_in_tensor_combiner_labels = 1
  clabel = tensor_combiner_labels[cpos_in_tensor_combiner_labels]
  c = combinedind(storage(tensor_combiner))
  labels_uc = deleteat(tensor_combiner_labels, cpos_in_tensor_combiner_labels)
  tensor_dest_labels = contract_labels(tensor_combiner_labels, tensor_src_labels)
  cpos_in_tensor_dest_labels = findfirst(==(clabel), tensor_dest_labels)
  tensor_dest_labels_uc = insertat(
    tensor_dest_labels, labels_uc, cpos_in_tensor_dest_labels
  )
  return contract_inds(
    inds(tensor_combiner),
    tensor_combiner_labels,
    inds(tensor_src),
    tensor_src_labels,
    tensor_dest_labels,
  )
end

# TODO: Move to `contract_uncombine.jl`.
function contract_uncombine_inds(
  tensor_src::Tensor{<:Any,<:Any,<:BlockSparseArray},
  tensor_src_labels,
  tensor_combiner::Tensor{<:Any,<:Any,<:CombinerArray},
  tensor_combiner_labels,
)
  return error("Not implemented")
end
