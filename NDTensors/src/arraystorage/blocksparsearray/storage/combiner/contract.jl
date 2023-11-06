function contract(a_src::BlockSparseArray, labels_src, a_comb::CombinerArray, labels_comb)
  # TODO: Special cases for index replacement, need
  # to check for trivial block permutations.
  return if is_combining(a_src, labels_src, a_comb, labels_comb)
    contract_combine(a_src, labels_src, a_comb, labels_comb)
  else
    # TODO: Check this is actually uncombining.
    contract_uncombine(a_src, labels_src, a_comb, labels_comb)
  end
end

function contract(a_comb::CombinerArray, labels_comb, a_src::BlockSparseArray, labels_src)
  return contract(a_src, labels_src, a_comb, labels_comb)
end
