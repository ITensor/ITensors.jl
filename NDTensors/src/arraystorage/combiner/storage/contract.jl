function contract(
  a_comb::CombinerArray, labels_comb, a_src::MatrixOrArrayStorage, labels_src
)
  if ndims(a_comb) ≤ 1
    return contract_scalar(a_comb, labels_comb, a_src, labels_src)
  elseif is_index_replacement(a_src, labels_src, a_comb, labels_comb)
    return contract_replacement(a_comb, labels_comb, a_src, labels_src)
  elseif is_combining(a_src, labels_src, a_comb, labels_comb)
    return contract_combine(a_comb, labels_comb, a_src, labels_src)
  else
    # TODO: Check this is a proper uncombining.
    return contract_uncombine(a_comb, labels_comb, a_src, labels_src)
  end
  return invalid_comb_contraction_error(t, labels_src, a_comb, labels_comb)
end

# Empty comb, acts as multiplying by 1
function contract_scalar(
  a_comb::CombinerArray, labels_comb, a_src::MatrixOrArrayStorage, labels_src
)
  error("Not implemented")
  return copy(a_src), labels_dest
end

function contract_replacement(
  a_comb::CombinerArray, labels_comb, a_src::MatrixOrArrayStorage, labels_src
)
  error("Not implemented")
  ui = setdiff(labels_comb, labels_src)[]
  new_axis = axes(a_comb)[findfirst(==(ui), labels_comb)]
  cpos1, cpos2 = intersect_positions(labels_comb, labels_src)
  a_dest = copy(a_src)
  ## axes_dest = setindex(axes(a_src), new_axis, cpos2)
  return a_dest, labels_dest
end

function contract_combine(
  a_comb::CombinerArray, labels_comb, a_src::MatrixOrArrayStorage, labels_src
)
  labels_dest = contract_labels(labels_comb, labels_src)
  axes_dest = contract_inds(axes(a_comb), labels_comb, axes(a_src), labels_src, labels_dest)
  cpos1, a_dest_cpos = intersect_positions(labels_comb, labels_dest)
  labels_comb = deleteat(labels_comb, cpos1)
  a_dest_vl = [labels_dest...]
  for (ii, li) in enumerate(labels_comb)
    insert!(a_dest_vl, a_dest_cpos + ii, li)
  end
  deleteat!(a_dest_vl, a_dest_cpos)
  labels_perm = tuple(a_dest_vl...)
  perm = getperm(labels_perm, labels_src)
  tp_axes = permute(axes(a_src), perm)
  a_dest = permutedims(a_src, perm)
  return reshape(a_dest, length.(axes_dest)), labels_dest
end

function contract_uncombine(
  a_comb::CombinerArray, labels_comb, a_src::MatrixOrArrayStorage, labels_src
)
  a_dest = copy(a_src)

  cpos1, cpos2 = intersect_positions(labels_comb, labels_src)

  axes_dest = deleteat(axes(a_comb), cpos1)
  axes_dest = insertat(axes(a_src), axes_dest, cpos2)

  labels_dest = deleteat(labels_comb, cpos1)
  labels_dest = insertat(labels_src, labels_dest, cpos2)

  return reshape(a_dest, length.(axes_dest)), labels_dest
end
