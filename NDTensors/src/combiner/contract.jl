function contraction_output(
  ::TensorT1, ::TensorT2, indsR::Tuple
) where {TensorT1<:CombinerTensor,TensorT2<:DenseTensor}
  TensorR = contraction_output_type(TensorT1, TensorT2, indsR)
  return similar(TensorR, indsR)
end

function contraction_output(
  T1::TensorT1, T2::TensorT2, indsR
) where {TensorT1<:DenseTensor,TensorT2<:CombinerTensor}
  return contraction_output(T2, T1, indsR)
end

function contract!!(
  output_tensor::Tensor,
  output_tensor_labels,
  combiner_tensor::CombinerTensor,
  combiner_tensor_labels,
  tensor::Tensor,
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
    output_tensor_inds = inds(output_tensor)
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
    tensorp = reshape(output_tensor, permute(inds(tensor), perm))
    permutedims!(tensorp, tensor, perm)
    return reshape(tensorp, output_tensor_inds)
  else # Uncombining
    cpos1, cpos2 = intersect_positions(combiner_tensor_labels, tensor_labels)
    output_tensor_storage = copy(storage(tensor))
    indsC = deleteat(inds(combiner_tensor), cpos1)
    output_tensor_inds = insertat(inds(tensor), indsC, cpos2)
    return NDTensors.tensor(output_tensor_storage, output_tensor_inds)
  end
  return invalid_combiner_contraction_error(
    tensor, tensor_labels, combiner_tensor, combiner_tensor_labels
  )
end

function contract!!(
  output_tensor::Tensor,
  output_tensor_labels,
  tensor::Tensor,
  tensor_labels,
  combiner_tensor::CombinerTensor,
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

function contract(
  diag_tensor::DiagTensor,
  diag_tensor_labels,
  combiner_tensor::CombinerTensor,
  combiner_tensor_labels,
)
  return contract(
    dense(diag_tensor), diag_tensor_labels, combiner_tensor, combiner_tensor_labels
  )
end
