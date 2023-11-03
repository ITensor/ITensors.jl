function contract_combine!!(
  ::ArrayStorage,
  ::Any,
  tensor::BlockSparseArray,
  tensor_labels,
  combiner_tensor::CombinerTensor,
  combiner_tensor_labels,
)
  # Get the label marking the combined index
  # By convention the combined index is the first one
  # TODO: Consider storing the location of the combined
  # index in preperation for multiple combined indices
  # TODO: Use `combinedind_label(...)`, `uncombinedind_labels(...)`, etc.
  cpos_in_combiner_tensor_labels = 1
  clabel = combiner_tensor_labels[cpos_in_combiner_tensor_labels]
  c = combinedind(combiner_tensor)
  labels_uc = deleteat(combiner_tensor_labels, cpos_in_combiner_tensor_labels)

  output_tensor_labels = contract_labels(combiner_tensor_labels, tensor_labels)
  cpos_in_output_tensor_labels = findfirst(==(clabel), output_tensor_labels)
  output_tensor_labels_uc = insertat(
    output_tensor_labels, labels_uc, cpos_in_output_tensor_labels
  )
  output_tensor_inds = contract_inds(
    inds(combiner_tensor),
    combiner_tensor_labels,
    inds(tensor),
    tensor_labels,
    output_tensor_labels,
  )

  #<fermions>:
  tensor = before_combiner_signs(
    tensor,
    tensor_labels,
    inds(tensor),
    combiner_tensor,
    combiner_tensor_labels,
    inds(combiner_tensor),
    output_tensor_labels,
    output_tensor_inds,
  )

  perm = getperm(output_tensor_labels_uc, tensor_labels)
  ucpos_in_tensor_labels = Tuple(findall(x -> x in labels_uc, tensor_labels))
  output_tensor = permutedims_combine(
    tensor,
    output_tensor_inds,
    perm,
    ucpos_in_tensor_labels,
    blockperm(combiner_tensor),
    blockcomb(combiner_tensor),
  )
  return output_tensor
end

# Return if the combiner contraction is combining or uncombining.
# Check for valid contractions, for example when combining,
# only the combined index should be uncontracted, and when uncombining,
# only the combined index should be contracted.
function is_combining(
  tensor::ArrayStorage, tensor_labels, combiner_tensor::CombinerTensor, combiner_tensor_labels
)
  is_combining = is_combining_no_check(
    tensor, tensor_labels, combiner_tensor, combiner_tensor_labels
  )
  check_valid_combiner_contraction(
    is_combining, tensor, tensor_labels, combiner_tensor, combiner_tensor_labels
  )
  return is_combining
end

function is_combining_no_check(
  tensor::ArrayStorage, tensor_labels, combiner_tensor::CombinerTensor, combiner_tensor_labels
)
  return combinedind_label(combiner_tensor, combiner_tensor_labels) ∉ tensor_labels
end

function check_valid_combiner_contraction(
  is_combining::Bool,
  tensor::ArrayStorage,
  tensor_labels,
  combiner_tensor::CombinerTensor,
  combiner_tensor_labels,
)
  if !is_valid_combiner_contraction(
    is_combining, tensor, tensor_labels, combiner_tensor, combiner_tensor_labels
  )
    return invalid_combiner_contraction_error(
      tensor, tensor_labels, combiner_tensor, combiner_tensor_labels
    )
  end
  return nothing
end

function is_valid_combiner_contraction(
  is_combining::Bool,
  tensor::ArrayStorage,
  tensor_labels,
  combiner_tensor::CombinerTensor,
  combiner_tensor_labels,
)
  in_tensor_labels_op = is_combining ? ∉(tensor_labels) : ∈(tensor_labels)
  return isone(count(in_tensor_labels_op, combiner_tensor_labels))
end
