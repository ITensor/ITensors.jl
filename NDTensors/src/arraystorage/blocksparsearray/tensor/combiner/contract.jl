function contract(
  t_src::Tensor{T,N,<:BlockSparseArray{T,N}},
  labels_src,
  t_comb::Tensor{Any,M,<:CombinerArray{M}},
  labels_comb,
) where {T,N,M}
  array_dest = contract(storage(t_src), labels_src, storage(t_comb), labels_comb)
  inds_dest = if is_combining(storage(t_src), labels_src, storage(t_comb), labels_comb)
    contract_combine_inds(t_src, labels_src, t_comb, labels_comb)
  else
    contract_uncombine_inds(t_src, labels_src, t_comb, labels_comb)
  end
  return tensor(array_dest, inds_dest)
end

function contract(
  t_comb::Tensor{Any,M,<:CombinerArray{M}},
  labels_comb,
  t_src::Tensor{T,N,<:BlockSparseArray{T,N}},
  labels_src,
) where {T,N,M}
  return contract(t_src, labels_src, t_comb, labels_comb)
end

function contract_combine_inds(
  t_src::Tensor{<:Any,<:Any,<:BlockSparseArray},
  labels_src,
  t_comb::Tensor{<:Any,<:Any,<:CombinerArray},
  labels_comb,
)

  # TODO: Not needed? Maybe for uncombining?
  # Get the label marking the combined index
  # By convention the combined index is the first one
  # TODO: Consider storing the location of the combined
  # index in preperation for multiple combined indices
  # TODO: Use `combinedind_label(...)`, `uncombinedind_labels(...)`, etc.
  ## cpos_in_labels_comb = 1
  ## clabel = labels_comb[cpos_in_labels_comb]
  ## c = combinedind(storage(t_comb))
  ## labels_uc = deleteat(labels_comb, cpos_in_labels_comb)
  ## cpos_in_labels_dest = findfirst(==(clabel), labels_dest)
  ## labels_dest_uc = insertat(
  ##   labels_dest, labels_uc, cpos_in_labels_dest
  ## )

  labels_dest = contract_labels(labels_comb, labels_src)
  return contract_inds(inds(t_comb), labels_comb, inds(t_src), labels_src, labels_dest)
end

function contract_uncombine_inds(
  t_src::Tensor{<:Any,<:Any,<:BlockSparseArray},
  labels_src,
  t_comb::Tensor{<:Any,<:Any,<:CombinerArray},
  labels_comb,
)
  return error("Not implemented")
end
