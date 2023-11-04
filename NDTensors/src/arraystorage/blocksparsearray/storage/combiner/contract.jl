function contract(
  a_src::BlockSparseArray,
  labels_src,
  a_comb::CombinerArray,
  labels_comb,
)
  return if is_combining(a_src, labels_src, a_comb, labels_comb)
    contract_combine(a_src, labels_src, a_comb, labels_comb)
  else
    contract_uncombine(a_src, labels_src, a_comb, labels_comb)
  end
end
