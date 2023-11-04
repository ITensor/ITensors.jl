function contraction_output(
  t_src::MatrixOrArrayStorageTensor, t_comb::Tensor{<:Any,<:Any,<:CombinerArray}, inds_dest
)
  t_dest_type = contraction_output_type(typeof(t_src), typeof(t_comb), inds_dest)
  return NDTensors.similar(t_dest_type, inds_dest)
end

function contract!!(
  t_dest::Tensor,
  labels_dest,
  t_comb::Tensor{<:Any,<:Any,<:CombinerArray},
  labels_comb,
  t_src::MatrixOrArrayStorageTensor,
  labels_src,
)
  a_dest = contract!!(
    storage(t_dest), labels_dest, storage(t_comb), labels_comb, storage(t_src), labels_src
  )
  inds_dest = contraction_output_inds(
    t_dest, labels_dest, t_comb, labels_comb, t_src, labels_src
  )
  return tensor(a_dest, inds_dest)
end

function contraction_output_inds(
  t_dest,
  labels_dest,
  t_comb::Tensor{<:Any,<:Any,<:CombinerArray},
  labels_comb,
  t_src,
  labels_src,
)
  # TODO: Define for scalar comb and replacement combiner contractions.
  return inds_dest = if ndims(t_comb) ≤ 1
    contraction_output_inds_scalar(
      t_dest, labels_dest, t_comb, labels_comb, t_src, labels_src
    )
  elseif is_index_replacement(storage(t_src), labels_src, storage(t_comb), labels_comb)
    contraction_output_inds_replacement(
      t_dest, labels_dest, t_comb, labels_comb, t_src, labels_src
    )
  elseif is_combining(storage(t_src), labels_src, storage(t_comb), labels_comb)
    contraction_output_inds_combining(
      t_dest, labels_dest, t_comb, labels_comb, t_src, labels_src
    )
  else
    contraction_output_inds_uncombining(
      t_dest, labels_dest, t_comb, labels_comb, t_src, labels_src
    )
  end
end

function contraction_output_inds_scalar(
  t_dest,
  labels_dest,
  t_comb::Tensor{<:Any,<:Any,<:CombinerArray},
  labels_comb,
  t_src,
  labels_src,
)
  return error("Not implemented")
end

function contraction_output_inds_replacement(
  t_dest,
  labels_dest,
  t_comb::Tensor{<:Any,<:Any,<:CombinerArray},
  labels_comb,
  t_src,
  labels_src,
)
  return error("Not implemented")
end

function contraction_output_inds_combining(
  t_dest,
  labels_dest,
  t_comb::Tensor{<:Any,<:Any,<:CombinerArray},
  labels_comb,
  t_src,
  labels_src,
)
  return inds(t_dest)
end

function contraction_output_inds_uncombining(
  t_dest,
  labels_dest,
  t_comb::Tensor{<:Any,<:Any,<:CombinerArray},
  labels_comb,
  t_src,
  labels_src,
)
  cpos1, cpos2 = intersect_positions(labels_comb, labels_src)
  indsC = deleteat(inds(t_comb), cpos1)
  return insertat(inds(t_src), indsC, cpos2)
end

function contract!!(
  t_dest::Tensor,
  labels_dest,
  t_src::MatrixOrArrayStorageTensor,
  labels_src,
  t_comb::Tensor{<:Any,<:Any,<:CombinerArray},
  labels_comb,
)
  return contract!!(t_dest, labels_dest, t_comb, labels_comb, t_src, labels_src)
end
