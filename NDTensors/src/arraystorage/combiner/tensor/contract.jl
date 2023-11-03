# Tensor definitions
function contraction_output(
  tensor1::MatrixOrArrayStorageTensor, tensor2::Tensor{<:Any,<:Any,<:CombinerArray}, indsR
)
  tensortypeR = contraction_output_type(typeof(tensor1), typeof(tensor2), indsR)
  return NDTensors.similar(tensortypeR, indsR)
end

# Tensor definitions
function contract!!(
  tensor_dest::Tensor,
  tensor_dest_labels,
  tensor_combiner::Tensor{<:Any,<:Any,<:CombinerArray},
  tensor_combiner_labels,
  tensor_src::MatrixOrArrayStorageTensor,
  tensor_src_labels,
)
  output_array = contract!!(
    storage(tensor_dest),
    tensor_dest_labels,
    storage(tensor_combiner),
    tensor_combiner_labels,
    storage(tensor_src),
    tensor_src_labels,
  )
  # TODO: Define for scalar combiner and replacement combiner contractions.
  tensor_dest_inds =
    if is_combining(
      storage(tensor_src),
      tensor_src_labels,
      storage(tensor_combiner),
      tensor_combiner_labels,
    )
      inds(tensor_dest)
    else
      cpos1, cpos2 = intersect_positions(tensor_combiner_labels, tensor_src_labels)
      indsC = deleteat(inds(tensor_combiner), cpos1)
      insertat(inds(tensor_src), indsC, cpos2)
    end
  return tensor(output_array, tensor_dest_inds)
end

# Tensor definitions
function contract!!(
  tensor_dest::Tensor,
  tensor_dest_labels,
  tensor_src::MatrixOrArrayStorageTensor,
  tensor_src_labels,
  tensor_combiner::Tensor{<:Any,<:Any,<:CombinerArray},
  tensor_combiner_labels,
)
  return contract!!(
    tensor_dest,
    tensor_dest_labels,
    tensor_combiner,
    tensor_combiner_labels,
    tensor_src,
    tensor_src_labels,
  )
end

# Storage definitions
# TODO: Split into multiple functions handling
# combining, uncombining, index replacement, etc.
function contract!!(
  tensor_dest::AbstractArray,
  tensor_dest_labels,
  tensor_combiner::CombinerArray,
  tensor_combiner_labels,
  tensor_src::MatrixOrArrayStorage,
  tensor_src_labels,
)
  if ndims(tensor_combiner) ≤ 1
    return contract_scalar!!(
      tensor_dest,
      tensor_dest_labels,
      tensor_combiner,
      tensor_combiner_labels,
      tensor_src,
      tensor_src_labels,
    )
  elseif is_index_replacement(
    tensor_src, tensor_src_labels, tensor_combiner, tensor_combiner_labels
  )
    return contract_replacement!!(
      tensor_dest,
      tensor_dest_labels,
      tensor_combiner,
      tensor_combiner_labels,
      tensor_src,
      tensor_src_labels,
    )
  elseif is_combining(
    tensor_src, tensor_src_labels, tensor_combiner, tensor_combiner_labels
  )
    return contract_combine!!(
      tensor_dest,
      tensor_dest_labels,
      tensor_combiner,
      tensor_combiner_labels,
      tensor_src,
      tensor_src_labels,
    )
  else
    return contract_uncombine!!(
      tensor_dest,
      tensor_dest_labels,
      tensor_combiner,
      tensor_combiner_labels,
      tensor_src,
      tensor_src_labels,
    )
  end
  return invalid_combiner_contraction_error(
    tensor, tensor_src_labels, tensor_combiner, tensor_combiner_labels
  )
end

# Empty combiner, acts as multiplying by 1
function contract_scalar!!(
  tensor_dest::AbstractArray,
  tensor_dest_labels,
  tensor_combiner::CombinerArray,
  tensor_combiner_labels,
  tensor_src::MatrixOrArrayStorage,
  tensor_src_labels,
)
  error("Not implemented")
  tensor_dest = permutedims!!(
    tensor_dest, tensor_src, getperm(tensor_dest_labels, tensor_src_labels)
  )
  return tensor_dest
end

function contract_replacement!!(
  tensor_dest::AbstractArray,
  tensor_dest_labels,
  tensor_combiner::CombinerArray,
  tensor_combiner_labels,
  tensor_src::MatrixOrArrayStorage,
  tensor_src_labels,
)
  error("Not implemented")
  ui = setdiff(tensor_combiner_labels, tensor_src_labels)[]
  newind = inds(tensor_combiner)[findfirst(==(ui), tensor_combiner_labels)]
  cpos1, cpos2 = intersect_positions(tensor_combiner_labels, tensor_src_labels)
  tensor_dest_storage = copy(storage(tensor_src))
  tensor_dest_inds = setindex(inds(tensor_src), newind, cpos2)
  return NDTensors.tensor(tensor_dest_storage, tensor_dest_inds)
end

function contract_combine!!(
  tensor_dest::AbstractArray,
  tensor_dest_labels,
  tensor_combiner::CombinerArray,
  tensor_combiner_labels,
  tensor_src::MatrixOrArrayStorage,
  tensor_src_labels,
)
  Alabels, Blabels = tensor_src_labels, tensor_combiner_labels
  final_labels = contract_labels(Blabels, Alabels)
  final_labels_n = contract_labels(tensor_combiner_labels, tensor_src_labels)
  tensor_dest_inds = axes(tensor_dest)
  if final_labels != final_labels_n
    perm = getperm(final_labels_n, final_labels)
    tensor_dest_inds = permute(inds(tensor_dest), perm)
    tensor_dest_labels = permute(tensor_dest_labels, perm)
  end
  cpos1, tensor_dest_cpos = intersect_positions(tensor_combiner_labels, tensor_dest_labels)
  labels_comb = deleteat(tensor_combiner_labels, cpos1)
  tensor_dest_vl = [tensor_dest_labels...]
  for (ii, li) in enumerate(labels_comb)
    insert!(tensor_dest_vl, tensor_dest_cpos + ii, li)
  end
  deleteat!(tensor_dest_vl, tensor_dest_cpos)
  labels_perm = tuple(tensor_dest_vl...)
  perm = getperm(labels_perm, tensor_src_labels)
  tensorp_inds = permute(axes(tensor_src), perm)
  tensorp = reshape(tensor_dest, length.(tensorp_inds))
  permutedims!(tensorp, tensor_src, perm)
  return reshape(tensorp, length.(tensor_dest_inds))
end

function contract_uncombine!!(
  tensor_dest::AbstractArray,
  tensor_dest_labels,
  tensor_combiner::CombinerArray,
  tensor_combiner_labels,
  tensor_src::MatrixOrArrayStorage,
  tensor_src_labels,
)
  cpos1, cpos2 = intersect_positions(tensor_combiner_labels, tensor_src_labels)
  tensor_dest_storage = copy(tensor_src)
  indsC = deleteat(axes(tensor_combiner), cpos1)
  tensor_dest_inds = insertat(axes(tensor_src), indsC, cpos2)
  return reshape(tensor_dest_storage, length.(tensor_dest_inds))
end
