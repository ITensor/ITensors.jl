blockperm(a::CombinerArray) = blockperm(a.combiner)
blockcomb(a::CombinerArray) = blockcomb(a.combiner)

function combinedind(a::CombinerArray)
  return axes(a)[combinedind_position(a)]
end

function is_index_replacement(
  a::AbstractArray, a_labels, a_comb::CombinerArray, a_comb_labels
)
  return (ndims(a_comb) == 2) && isone(count(∈(a_labels), a_comb_labels))
end

# Return if the combiner contraction is combining or uncombining.
# Check for valid contractions, for example when combining,
# only the combined index should be uncontracted, and when uncombining,
# only the combined index should be contracted.
function is_combining(a::AbstractArray, a_labels, a_comb::CombinerArray, a_comb_labels)
  is_combining = is_combining_no_check(a, a_labels, a_comb, a_comb_labels)
  check_valid_combiner_contraction(is_combining, a, a_labels, a_comb, a_comb_labels)
  return is_combining
end

function is_combining_no_check(
  a::AbstractArray, a_labels, a_comb::CombinerArray, a_comb_labels
)
  return combinedind_label(a_comb, a_comb_labels) ∉ a_labels
end

function combinedind_label(a_comb::CombinerArray, a_comb_labels)
  return a_comb_labels[combinedind_position(a_comb)]
end

# The position of the combined index/dimension.
# By convention, it is the first one.
combinedind_position(a_comb::CombinerArray) = 1

function check_valid_combiner_contraction(
  is_combining::Bool, a::AbstractArray, a_labels, a_comb::CombinerArray, a_comb_labels
)
  if !is_valid_combiner_contraction(is_combining, a, a_labels, a_comb, a_comb_labels)
    return invalid_combiner_contraction_error(a, a_labels, a_comb, a_comb_labels)
  end
  return nothing
end

function is_valid_combiner_contraction(
  is_combining::Bool, a::AbstractArray, a_labels, a_comb::CombinerArray, a_comb_labels
)
  in_a_labels_op = is_combining ? ∉(a_labels) : ∈(a_labels)
  return isone(count(in_a_labels_op, a_comb_labels))
end
