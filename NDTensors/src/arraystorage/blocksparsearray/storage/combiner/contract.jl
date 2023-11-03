function contract(
  tensor::BlockSparseArray,
  tensor_labels,
  combiner_tensor::CombinerArray,
  combiner_tensor_labels,
)
  if is_combining(tensor, tensor_labels, combiner_tensor, combiner_tensor_labels)
    return contract_combine(tensor, tensor_labels, combiner_tensor, combiner_tensor_labels)
  else # Uncombining
    return contract_uncombine(
      tensor, tensor_labels, combiner_tensor, combiner_tensor_labels
    )
  end
  return invalid_combiner_contraction_error(
    combiner_tensor, tensor_labels, tensor, tensor_labels
  )
end
