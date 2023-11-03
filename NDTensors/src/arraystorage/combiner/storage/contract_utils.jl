blockperm(C::CombinerArray) = blockperm(C.combiner)
blockcomb(C::CombinerArray) = blockcomb(C.combiner)

function combinedind(combiner_tensor::CombinerArray)
  return axes(combiner_tensor)[combinedind_position(combiner_tensor)]
end

function is_index_replacement(
  tensor::AbstractArray,
  tensor_labels,
  combiner_tensor::CombinerArray,
  combiner_tensor_labels,
)
  return (ndims(combiner_tensor) == 2) &&
         isone(count(∈(tensor_labels), combiner_tensor_labels))
end

# Return if the combiner contraction is combining or uncombining.
# Check for valid contractions, for example when combining,
# only the combined index should be uncontracted, and when uncombining,
# only the combined index should be contracted.
function is_combining(
  tensor::AbstractArray,
  tensor_labels,
  combiner_tensor::CombinerArray,
  combiner_tensor_labels,
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
  tensor::AbstractArray,
  tensor_labels,
  combiner_tensor::CombinerArray,
  combiner_tensor_labels,
)
  return combinedind_label(combiner_tensor, combiner_tensor_labels) ∉ tensor_labels
end

function combinedind_label(combiner_tensor::CombinerArray, combiner_tensor_labels)
  return combiner_tensor_labels[combinedind_position(combiner_tensor)]
end

# The position of the combined index/dimension.
# By convention, it is the first one.
combinedind_position(combiner_tensor::CombinerArray) = 1

function check_valid_combiner_contraction(
  is_combining::Bool,
  tensor::AbstractArray,
  tensor_labels,
  combiner_tensor::CombinerArray,
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
  tensor::AbstractArray,
  tensor_labels,
  combiner_tensor::CombinerArray,
  combiner_tensor_labels,
)
  in_tensor_labels_op = is_combining ? ∉(tensor_labels) : ∈(tensor_labels)
  return isone(count(in_tensor_labels_op, combiner_tensor_labels))
end
