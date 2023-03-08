#<fermions>:
function before_combiner_signs(
  tensor,
  tensor_labels,
  indstensor,
  combiner_tensor,
  combiner_tensor_labels,
  indscombiner_tensor,
  labelsoutput_tensor,
  output_tensor_inds,
)
  return tensor
end
function after_combiner_signs(
  output_tensor,
  labelsoutput_tensor,
  output_tensor_inds,
  combiner_tensor,
  combiner_tensor_labels,
  indscombiner_tensor,
)
  return output_tensor
end

function contract(
  tensor::BlockSparseTensor,
  tensor_labels,
  combiner_tensor::CombinerTensor,
  combiner_tensor_labels,
)
  #@timeit_debug timer "Block sparse (un)combiner" begin
  # Get the label marking the combined index
  # By convention the combined index is the first one
  # TODO: Consider storing the location of the combined
  # index in preperation for multiple combined indices
  # TODO: Use `combinedind_label(...)`, `uncombinedind_labels(...)`, etc.
  cpos_in_combiner_tensor_labels = 1
  clabel = combiner_tensor_labels[cpos_in_combiner_tensor_labels]
  c = combinedind(combiner_tensor)
  labels_uc = deleteat(combiner_tensor_labels, cpos_in_combiner_tensor_labels)
  is_combining_contraction = is_combining(
    tensor, tensor_labels, combiner_tensor, combiner_tensor_labels
  )
  if is_combining_contraction
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
  else # Uncombining
    output_tensor_labels = tensor_labels
    cpos_in_output_tensor_labels = findfirst(==(clabel), output_tensor_labels)
    # Move combined index to first position
    if cpos_in_output_tensor_labels != 1
      output_tensor_labels_orig = output_tensor_labels
      output_tensor_labels = deleteat(output_tensor_labels, cpos_in_output_tensor_labels)
      output_tensor_labels = insertafter(output_tensor_labels, clabel, 0)
      cpos_in_output_tensor_labels = 1
      perm = getperm(output_tensor_labels, output_tensor_labels_orig)
      tensor = permutedims(tensor, perm)
      tensor_labels = permute(tensor_labels, perm)
    end
    output_tensor_labels_uc = insertat(
      output_tensor_labels, labels_uc, cpos_in_output_tensor_labels
    )
    output_tensor_inds_uc = contract_inds(
      inds(combiner_tensor),
      combiner_tensor_labels,
      inds(tensor),
      tensor_labels,
      output_tensor_labels_uc,
    )

    # <fermions>:
    tensor = before_combiner_signs(
      tensor,
      tensor_labels,
      inds(tensor),
      combiner_tensor,
      combiner_tensor_labels,
      inds(combiner_tensor),
      output_tensor_labels_uc,
      output_tensor_inds_uc,
    )

    output_tensor = uncombine(
      tensor,
      tensor_labels,
      output_tensor_inds_uc,
      output_tensor_labels_uc,
      cpos_in_output_tensor_labels,
      blockperm(combiner_tensor),
      blockcomb(combiner_tensor),
    )

    # <fermions>:
    output_tensor = after_combiner_signs(
      output_tensor,
      output_tensor_labels_uc,
      output_tensor_inds_uc,
      combiner_tensor,
      combiner_tensor_labels,
      inds(combiner_tensor),
    )

    return output_tensor
  end
  return invalid_combiner_contraction_error(
    combiner_tensor, tensor_labels, tensor, tensor_labels
  )
end

function contract(
  combiner_tensor::CombinerTensor,
  combiner_tensor_labels,
  tensor::BlockSparseTensor,
  tensor_labels,
)
  return contract(tensor, tensor_labels, combiner_tensor, combiner_tensor_labels)
end

# Special case when no indices are combined
# TODO: No copy? Maybe use `AllowAlias`.
function contract(
  tensor::BlockSparseTensor,
  tensor_labels,
  combiner_tensor::CombinerTensor{<:Any,0},
  combiner_tensor_labels,
)
  return copy(tensor)
end
