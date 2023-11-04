function contract(
  t_comb::Tensor{<:Any,<:Any,<:CombinerArray},
  labels_comb,
  t_src::MatrixOrArrayStorageTensor,
  labels_src,
)
  a_dest = contract(storage(t_comb), labels_comb, storage(t_src), labels_src)
  inds_dest = contraction_output_inds(t_comb, labels_comb, t_src, labels_src)
  return tensor(a_dest, inds_dest)
end

function contraction_output_inds(
  t_comb::Tensor{<:Any,<:Any,<:CombinerArray}, labels_comb, t_src, labels_src
)
  return if ndims(t_comb) ≤ 1
    contraction_output_inds_scalar(t_comb, labels_comb, t_src, labels_src)
  elseif is_index_replacement(storage(t_src), labels_src, storage(t_comb), labels_comb)
    contraction_output_inds_replacement(t_comb, labels_comb, t_src, labels_src)
  elseif is_combining(storage(t_src), labels_src, storage(t_comb), labels_comb)
    contraction_output_inds_combining(t_comb, labels_comb, t_src, labels_src)
  else
    contraction_output_inds_uncombining(t_comb, labels_comb, t_src, labels_src)
  end
end

function contraction_output_inds_scalar(
  t_comb::Tensor{<:Any,<:Any,<:CombinerArray}, labels_comb, t_src, labels_src
)
  return error("Not implemented")
end

function contraction_output_inds_replacement(
  t_comb::Tensor{<:Any,<:Any,<:CombinerArray}, labels_comb, t_src, labels_src
)
  return error("Not implemented")
end

function contraction_output_inds_combining(
  t_comb::Tensor{<:Any,<:Any,<:CombinerArray}, labels_comb, t_src, labels_src
)
  labels_dest = contract_labels(labels_comb, labels_src)
  return contract_inds(inds(t_comb), labels_comb, inds(t_src), labels_src, labels_dest)
end

function contraction_output_inds_uncombining(
  t_comb::Tensor{<:Any,<:Any,<:CombinerArray}, labels_comb, t_src, labels_src
)
  cpos1, cpos2 = intersect_positions(labels_comb, labels_src)
  indsC = deleteat(inds(t_comb), cpos1)
  return insertat(inds(t_src), indsC, cpos2)
end

function contract(
  t_src::MatrixOrArrayStorageTensor,
  labels_src,
  t_comb::Tensor{<:Any,<:Any,<:CombinerArray},
  labels_comb,
)
  return contract(t_comb, labels_comb, t_src, labels_src)
end
