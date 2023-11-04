function contract!!(
  a_dest::AbstractArray,
  labels_dest,
  a_comb::CombinerArray,
  labels_comb,
  a_src::MatrixOrArrayStorage,
  labels_src,
)
  if ndims(a_comb) ≤ 1
    return contract_scalar!!(a_dest, labels_dest, a_comb, labels_comb, a_src, labels_src)
  elseif is_index_replacement(a_src, labels_src, a_comb, labels_comb)
    return contract_replacement!!(
      a_dest, labels_dest, a_comb, labels_comb, a_src, labels_src
    )
  elseif is_combining(a_src, labels_src, a_comb, labels_comb)
    return contraca_combine!!(a_dest, labels_dest, a_comb, labels_comb, a_src, labels_src)
  else
    return contract_uncombine!!(a_dest, labels_dest, a_comb, labels_comb, a_src, labels_src)
  end
  return invalid_comb_contraction_error(t, labels_src, a_comb, labels_comb)
end

# Empty comb, acts as multiplying by 1
function contract_scalar!!(
  a_dest::AbstractArray,
  labels_dest,
  a_comb::CombinerArray,
  labels_comb,
  a_src::MatrixOrArrayStorage,
  labels_src,
)
  error("Not implemented")
  a_dest = permutedims!!(a_dest, a_src, getperm(labels_dest, labels_src))
  return a_dest
end

function contract_replacement!!(
  a_dest::AbstractArray,
  labels_dest,
  a_comb::CombinerArray,
  labels_comb,
  a_src::MatrixOrArrayStorage,
  labels_src,
)
  error("Not implemented")
  ui = setdiff(labels_comb, labels_src)[]
  newind = inds(a_comb)[findfirst(==(ui), labels_comb)]
  cpos1, cpos2 = intersect_positions(labels_comb, labels_src)
  a_dest = copy(storage(a_src))
  inds_dest = setindex(inds(a_src), newind, cpos2)
  return tensor(a_dest, inds_dest)
end

function contraca_combine!!(
  a_dest::AbstractArray,
  labels_dest,
  a_comb::CombinerArray,
  labels_comb,
  a_src::MatrixOrArrayStorage,
  labels_src,
)
  Alabels, Blabels = labels_src, labels_comb
  final_labels = contract_labels(Blabels, Alabels)
  final_labels_n = contract_labels(labels_comb, labels_src)
  inds_dest = axes(a_dest)
  if final_labels != final_labels_n
    perm = getperm(final_labels_n, final_labels)
    inds_dest = permute(inds(a_dest), perm)
    labels_dest = permute(labels_dest, perm)
  end
  cpos1, a_dest_cpos = intersect_positions(labels_comb, labels_dest)
  labels_comb = deleteat(labels_comb, cpos1)
  a_dest_vl = [labels_dest...]
  for (ii, li) in enumerate(labels_comb)
    insert!(a_dest_vl, a_dest_cpos + ii, li)
  end
  deleteat!(a_dest_vl, a_dest_cpos)
  labels_perm = tuple(a_dest_vl...)
  perm = getperm(labels_perm, labels_src)
  tp_inds = permute(axes(a_src), perm)
  tp = reshape(a_dest, length.(tp_inds))
  permutedims!(tp, a_src, perm)
  return reshape(tp, length.(inds_dest))
end

function contract_uncombine!!(
  a_dest::AbstractArray,
  labels_dest,
  a_comb::CombinerArray,
  labels_comb,
  a_src::MatrixOrArrayStorage,
  labels_src,
)
  cpos1, cpos2 = intersect_positions(labels_comb, labels_src)
  a_dest = copy(a_src)
  indsC = deleteat(axes(a_comb), cpos1)
  inds_dest = insertat(axes(a_src), indsC, cpos2)
  return reshape(a_dest, length.(inds_dest))
end
