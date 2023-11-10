function key_dest(labels_dest, I1::CartesianIndex, labels1, I2::CartesianIndex, labels2)
  i_dest = ntuple(length(labels_dest)) do j_dest
    label_dest = labels_dest[j_dest]
    j1 = findfirst(==(label_dest), labels1)
    if !isnothing(j1)
      return Tuple(I1)[j1]
    end
    j2 = findfirst(==(label_dest), labels2)
    if !isnothing(j2)
      return Tuple(I2)[j2]
    end
    return nothing
  end
  if any(isnothing, i_dest)
    return nothing
  end
  return CartesianIndex(i_dest)
end

function default_contract_muladd(a1, labels1, a2, labels2, a_dest, labels_dest)
  return muladd(a1, a2, a_dest)
end

function contract!(
  a_dest::SparseArray,
  labels_dest,
  a1::SparseArray,
  labels1,
  a2::SparseArray,
  labels2;
  muladd=default_contract_muladd,
)
  for I1 in nonzero_keys(a1)
    for I2 in nonzero_keys(a2)
      # TODO: Cache information needed for `key_dest`, i.e.
      # location of `labels_dest` in `labels1` and `labels2`.
      I_dest = key_dest(labels_dest, I1, labels1, I2, labels2)
      if !isnothing(I_dest)
        # TODO: Use `contract!!` once we change to `UnallocatedZeros`
        # for structural zero blocks.
        a_dest[I_dest] = muladd(a1[I1], labels1, a2[I2], labels2, a_dest[I_dest], labels_dest)
      end
    end
  end
  return a_dest
end

function blocksparse_contract_muladd(a1, labels1, a2, labels2, a_dest, labels_dest)
  # TODO: Check that `α` and `β` are correct.
  contract!(a_dest, labels_dest, a1, labels1, a2, labels2, true, true)
  return a_dest
end

function contract!(
  a_dest::BlockSparseArray,
  labels_dest,
  a1::BlockSparseArray,
  labels1,
  a2::BlockSparseArray,
  labels2,
)
  contract!(blocks(a_dest), labels_dest, blocks(a1), labels1, blocks(a2), labels2; muladd=blocksparse_contract_muladd)
  return a_dest
end

function contract(
  a1::BlockSparseArray,
  labels1,
  a2::BlockSparseArray,
  labels2,
)
  labels_dest = contract_labels(labels1, labels2)
  axes_dest = contract_inds(axes(a1), labels1, axes(a2), labels2, labels_dest)
  # TODO: Do this through `allocate_output(::typeof(contract), ...)`
  elt_dest = promote_type(eltype(a1), eltype(a2))
  a_dest = BlockSparseArray{elt_dest}(axes_dest)
  contract!(a_dest, labels_dest, a1, labels1, a2, labels2)
  return a_dest, labels_dest
end

function contract(
  a1::BlockSparseArray,
  labels1,
  a2::Array,
  labels2,
)
  return error("Not implemented")
end

function contract(
  a1::Array,
  labels1,
  a2::BlockSparseArray,
  labels2,
)
  return error("Not implemented")
end
