# Tensor definitions
function contraction_output(
  tensor1::MatrixOrArrayStorageTensor, tensor2::Tensor{<:Any,<:Any,<:CombinerArray}, indsR
)
  tensortypeR = contraction_output_type(typeof(tensor1), typeof(tensor2), indsR)
  return NDTensors.similar(tensortypeR, indsR)
end

# Tensor definitions
function contract!!(
  output_tensor::Tensor,
  output_tensor_labels,
  combiner_tensor::Tensor{<:Any,<:Any,<:CombinerArray},
  combiner_tensor_labels,
  tensor_src::MatrixOrArrayStorageTensor,
  tensor_src_labels,
)
  output_array = contract!!(
    storage(output_tensor),
    output_tensor_labels,
    storage(combiner_tensor),
    combiner_tensor_labels,
    storage(tensor_src),
    tensor_src_labels,
  )

  # TODO: The Index ordering is probably wrong for some combining operations.
  return tensor(output_array, inds(output_tensor))
end

# Tensor definitions
function contract!!(
  output_tensor::Tensor,
  output_tensor_labels,
  tensor::MatrixOrArrayStorageTensor,
  tensor_labels,
  combiner_tensor::Tensor{<:Any,<:Any,<:CombinerArray},
  combiner_tensor_labels,
)
  return contract!!(
    output_tensor,
    output_tensor_labels,
    combiner_tensor,
    combiner_tensor_labels,
    tensor,
    tensor_labels,
  )
end

# Storage definitions
# TODO: Split into multiple functions handling
# combining, uncombining, index replacement, etc.
function contract!!(
  output_tensor::AbstractArray,
  output_tensor_labels,
  combiner_tensor::CombinerArray,
  combiner_tensor_labels,
  tensor::MatrixOrArrayStorage,
  tensor_labels,
)
  if ndims(combiner_tensor) ≤ 1
    # Empty combiner, acts as multiplying by 1
    output_tensor = permutedims!!(
      output_tensor, tensor, getperm(output_tensor_labels, tensor_labels)
    )
    return output_tensor
  end
  if is_index_replacement(tensor, tensor_labels, combiner_tensor, combiner_tensor_labels)
    ui = setdiff(combiner_tensor_labels, tensor_labels)[]
    newind = inds(combiner_tensor)[findfirst(==(ui), combiner_tensor_labels)]
    cpos1, cpos2 = intersect_positions(combiner_tensor_labels, tensor_labels)
    output_tensor_storage = copy(storage(tensor))
    output_tensor_inds = setindex(inds(tensor), newind, cpos2)
    return NDTensors.tensor(output_tensor_storage, output_tensor_inds)
  end
  is_combining_contraction = is_combining(
    tensor, tensor_labels, combiner_tensor, combiner_tensor_labels
  )
  if is_combining_contraction
    Alabels, Blabels = tensor_labels, combiner_tensor_labels
    final_labels = contract_labels(Blabels, Alabels)
    final_labels_n = contract_labels(combiner_tensor_labels, tensor_labels)
    output_tensor_inds = axes(output_tensor)
    if final_labels != final_labels_n
      perm = getperm(final_labels_n, final_labels)
      output_tensor_inds = permute(inds(output_tensor), perm)
      output_tensor_labels = permute(output_tensor_labels, perm)
    end
    cpos1, output_tensor_cpos = intersect_positions(
      combiner_tensor_labels, output_tensor_labels
    )
    labels_comb = deleteat(combiner_tensor_labels, cpos1)
    output_tensor_vl = [output_tensor_labels...]
    for (ii, li) in enumerate(labels_comb)
      insert!(output_tensor_vl, output_tensor_cpos + ii, li)
    end
    deleteat!(output_tensor_vl, output_tensor_cpos)
    labels_perm = tuple(output_tensor_vl...)
    perm = getperm(labels_perm, tensor_labels)
    tensorp_inds = permute(axes(tensor), perm)
    tensorp = reshape(output_tensor, length.(tensorp_inds))
    permutedims!(tensorp, tensor, perm)
    return reshape(tensorp, length.(output_tensor_inds))
  else # Uncombining
    cpos1, cpos2 = intersect_positions(combiner_tensor_labels, tensor_labels)
    output_tensor_storage = copy(storage(tensor))
    indsC = deleteat(inds(combiner_tensor), cpos1)
    output_tensor_inds = insertat(inds(tensor), indsC, cpos2)
    error("Not implemented")
    # TODO: Add a `reshape` for `ArrayStorageTensor`.
    return NDTensors.tensor(
      reshape(output_tensor_storage, dims(output_tensor_inds)), output_tensor_inds
    )
  end
  return invalid_combiner_contraction_error(
    tensor, tensor_labels, combiner_tensor, combiner_tensor_labels
  )
end
