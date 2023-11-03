function contract!!(
  tensor_dest::ArrayStorage,
  tensor_dest_labels::Any,
  tensor::BlockSparseArray,
  tensor_labels,
  combiner_tensor::CombinerTensor,
  combiner_tensor_labels,
)
  is_combining_contraction = is_combining(
    tensor, tensor_labels, combiner_tensor, combiner_tensor_labels
  )
  if is_combining_contraction
    return contract_combine!!(tensor_dest, tensor_dest_labels, tensor, tensor_labels, combiner_tensor, combiner_tensor_labels)
  else # Uncombining
    return contract_uncombine!!(tensor_dest, tensor_dest_labels, tensor, tensor_labels, combiner_tensor, combiner_tensor_labels)
  end
  return invalid_combiner_contraction_error(
    combiner_tensor, tensor_labels, tensor, tensor_labels
  )
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
