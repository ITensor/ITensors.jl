function contract_combine(
  a_src::BlockSparseArray, labels_src, a_comb::CombinerArray, labels_comb
)
  labels_dest = contract_labels(labels_comb, labels_src)
  axes_dest = contract_inds(axes(a_comb), labels_comb, axes(a_src), labels_src, labels_dest)

  ## TODO: Add this back.
  ## #<fermions>:
  ## a_src = before_combiner_signs(
  ##   a_src,
  ##   labels_src,
  ##   axes(a_src),
  ##   a_comb,
  ##   labels_comb,
  ##   axes(a_comb),
  ##   labels_dest,
  ##   axes_dest,
  ## )

  # Account for permutation of data.
  cpos_in_labels_comb = 1
  clabel = labels_comb[cpos_in_labels_comb]
  labels_uc = deleteat(labels_comb, cpos_in_labels_comb)
  cpos_in_labels_dest = findfirst(==(clabel), labels_dest)
  labels_dest_uc = insertat(labels_dest, labels_uc, cpos_in_labels_dest)
  perm = getperm(labels_dest_uc, labels_src)
  ucpos_in_labels_src = Tuple(findall(x -> x in labels_uc, labels_src))
  a_dest = permutedims_combine(
    a_src, axes_dest, perm, ucpos_in_labels_src, blockperm(a_comb), blockcomb(a_comb)
  )

  return a_dest, labels_dest
end

function permutedims_combine(
  a_src::BlockSparseArray,
  axes_dest,
  perm::Tuple,
  combdims::Tuple,
  blockperm::Vector{Int},
  blockcomb::Vector{Int},
)
  a_dest = permutedims_combine_output(
    a_src, axes_dest, perm, combdims, blockperm, blockcomb
  )

  # Permute the indices
  axes_perm = permute(axes(a_src), perm)

  # Now that the indices are permuted, compute
  # which indices are now combined
  combdims_perm = TupleTools.sort(_permute_combdims(combdims, perm))
  comb_ind_loc = minimum(combdims_perm)

  # Determine the new index before combining
  axes_to_combine = getindices(axes_perm, combdims_perm)
  axis_comb = ⊗(axes_to_combine...)
  axis_comb = BlockArrays.blockedrange(length.(BlockArrays.blocks(axis_comb)[blockperm]))

  for b in nzblocks(a_src)
    a_src_b = @view a_src[b]
    b_perm = permute(b, perm)
    b_perm_comb = combine_dims(b_perm, axes_perm, combdims_perm)
    b_perm_comb = perm_block(b_perm_comb, comb_ind_loc, blockperm)
    # TODO: Wrap this in `BlockArrays.Block`?
    b_in_combined_dim = b_perm_comb.n[comb_ind_loc]
    new_b_in_combined_dim = blockcomb[b_in_combined_dim]
    offset = 0
    pos_in_new_combined_block = 1
    while b_in_combined_dim - pos_in_new_combined_block > 0 &&
      blockcomb[b_in_combined_dim - pos_in_new_combined_block] == new_b_in_combined_dim
      # offset += blockdim(axis_comb, b_in_combined_dim - pos_in_new_combined_block)
      offset += length(
        axis_comb[BlockArrays.Block(b_in_combined_dim - pos_in_new_combined_block)]
      )
      pos_in_new_combined_block += 1
    end
    b_dest = setindex(b_perm_comb, new_b_in_combined_dim, comb_ind_loc)
    a_dest_b_total = @view a_dest[b_dest]
    # dimsa_dest_b_tot = size(a_dest_b_total)

    # TODO: Simplify this code.
    subind = ntuple(ndims(a_src) - length(combdims) + 1) do i
      if i == comb_ind_loc
        range(
          1 + offset;
          stop=offset + length(axis_comb[BlockArrays.Block(b_in_combined_dim)]),
        )
      else
        range(1; stop=size(a_dest_b_total)[i])
      end
    end

    a_dest_b = @view a_dest_b_total[subind...]
    a_dest_b = reshape(a_dest_b, permute(size(a_src_b), perm))
    # TODO: Make this `convert` call more general
    # for GPUs using `unwrap_array_type`.
    a_src_bₐ = convert(Array, a_src_b)
    # TODO: Use `expose` to make more efficient and robust.
    permutedims!(a_dest_b, a_src_bₐ, perm)
  end

  return a_dest
end

function permutedims_combine_output(
  a_src::BlockSparseArray,
  axes_dest,
  perm::Tuple,
  combdims::Tuple,
  blockperm::Vector{Int},
  blockcomb::Vector{Int},
)
  # Permute the indices
  axes_src = axes(a_src)
  axes_perm = permute(axes_src, perm)

  # Now that the indices are permuted, compute
  # which indices are now combined
  combdims_perm = TupleTools.sort(_permute_combdims(combdims, perm))

  # Permute the nonzero blocks (dimension-wise)
  blocks = nzblocks(a_src)

  # TODO: Use `permute.(blocks, perm)`.
  blocks_perm = BlockArrays.Block.(permute.(getfield.(blocks, :n), Ref(perm)))

  # Combine the nonzero blocks (dimension-wise)
  blocks_perm_comb = combine_dims(blocks_perm, axes_perm, combdims_perm)

  # Permute the blocks (within the newly combined dimension)
  comb_ind_loc = minimum(combdims_perm)
  blocks_perm_comb = perm_blocks(blocks_perm_comb, comb_ind_loc, blockperm)
  blocks_perm_comb = sort(blocks_perm_comb; lt=isblockless)

  # Combine the blocks (within the newly combined and permuted dimension)
  blocks_perm_comb = combine_blocks(blocks_perm_comb, comb_ind_loc, blockcomb)
  T = eltype(a_src)
  N = length(axes_dest)
  B = set_ndims(unwrap_array_type(a_src), length(axes_dest))
  return BlockSparseArray{T,N,B}(undef, blocks_perm_comb, axes_dest)
end
