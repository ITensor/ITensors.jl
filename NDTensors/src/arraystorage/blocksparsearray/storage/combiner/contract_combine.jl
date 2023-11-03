function contract_combine(
  tensor::BlockSparseArray,
  tensor_labels,
  combiner_tensor::CombinerArray,
  combiner_tensor_labels,
)
  # Get the label marking the combined index
  # By convention the combined index is the first one
  # TODO: Consider storing the location of the combined
  # index in preperation for multiple combined indices
  # TODO: Use `combinedind_label(...)`, `uncombinedind_labels(...)`, etc.
  cpos_in_combiner_tensor_labels = 1
  clabel = combiner_tensor_labels[cpos_in_combiner_tensor_labels]
  c = combinedind(combiner_tensor)
  labels_uc = deleteat(combiner_tensor_labels, cpos_in_combiner_tensor_labels)

  output_tensor_labels = contract_labels(combiner_tensor_labels, tensor_labels)
  cpos_in_output_tensor_labels = findfirst(==(clabel), output_tensor_labels)
  output_tensor_labels_uc = insertat(
    output_tensor_labels, labels_uc, cpos_in_output_tensor_labels
  )

  # TODO: This is doing the wrong thing for partial combining.
  # Rewrite so that we don't need this.
  output_tensor_inds = contract_inds(
    axes(combiner_tensor),
    combiner_tensor_labels,
    axes(tensor),
    tensor_labels,
    output_tensor_labels,
  )

  ## TODO: Add this back.
  ## #<fermions>:
  ## tensor = before_combiner_signs(
  ##   tensor,
  ##   tensor_labels,
  ##   inds(tensor),
  ##   combiner_tensor,
  ##   combiner_tensor_labels,
  ##   inds(combiner_tensor),
  ##   output_tensor_labels,
  ##   output_tensor_inds,
  ## )

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
end

function permutedims_combine(
  T::BlockSparseArray{ElT,N},
  is,
  perm::NTuple{N,Int},
  combdims::NTuple{NC,Int},
  blockperm::Vector{Int},
  blockcomb::Vector{Int},
) where {ElT,N,NC}
  R = permutedims_combine_output(T, is, perm, combdims, blockperm, blockcomb)

  # Permute the indices
  inds_perm = permute(axes(T), perm)

  # Now that the indices are permuted, compute
  # which indices are now combined
  combdims_perm = sort(_permute_combdims(combdims, perm))
  comb_ind_loc = minimum(combdims_perm)

  # Determine the new index before combining
  inds_to_combine = getindices(inds_perm, combdims_perm)
  ind_comb = ⊗(inds_to_combine...)
  ## ind_comb = permuteblocks(ind_comb, blockperm)
  ind_comb = BlockArrays.blockedrange(length.(BlockArrays.blocks(ind_comb)[blockperm]))

  for b in nzblocks(T)
    Tb = @view T[b]
    b_perm = permute(b, perm)
    b_perm_comb = combine_dims(b_perm, inds_perm, combdims_perm)
    b_perm_comb = perm_block(b_perm_comb, comb_ind_loc, blockperm)
    # TODO: Wrap this in `BlockArrays.Block`?
    b_in_combined_dim = b_perm_comb.n[comb_ind_loc]
    new_b_in_combined_dim = blockcomb[b_in_combined_dim]
    offset = 0
    pos_in_new_combined_block = 1
    while b_in_combined_dim - pos_in_new_combined_block > 0 &&
      blockcomb[b_in_combined_dim - pos_in_new_combined_block] == new_b_in_combined_dim
      # offset += blockdim(ind_comb, b_in_combined_dim - pos_in_new_combined_block)
      offset += length(
        ind_comb[BlockArrays.Block(b_in_combined_dim - pos_in_new_combined_block)]
      )
      pos_in_new_combined_block += 1
    end
    b_new = setindex(b_perm_comb, new_b_in_combined_dim, comb_ind_loc)
    Rb_total = @view R[b_new]
    dimsRb_tot = size(Rb_total)
    subind = ntuple(
      i -> if i == comb_ind_loc
        range(
          1 + offset; stop=offset + length(ind_comb[BlockArrays.Block(b_in_combined_dim)])
        )
      else
        range(1; stop=dimsRb_tot[i])
      end,
      N - NC + 1,
    )
    Rb = @view Rb_total[subind...]
    # XXX Are these equivalent?
    #Tb_perm = permutedims(Tb,perm)
    #copyto!(Rb,Tb_perm)

    Rb = reshape(Rb, permute(size(Tb), perm))
    # TODO: Make this `convert` call more general
    # for GPUs.
    Tbₐ = convert(Array, Tb)
    ## @strided Rb .= permutedims(Tbₐ, perm)
    permutedims!(Rb, Tbₐ, perm)
  end

  return R
end

function permutedims_combine_output(
  T::BlockSparseArray{ElT,N},
  is,
  perm::NTuple{N,Int},
  combdims::NTuple{NC,Int},
  blockperm::Vector{Int},
  blockcomb::Vector{Int},
) where {ElT,N,NC}
  # Permute the indices
  indsT = axes(T)
  inds_perm = permute(indsT, perm)

  # Now that the indices are permuted, compute
  # which indices are now combined
  combdims_perm = sort(_permute_combdims(combdims, perm))

  # Permute the nonzero blocks (dimension-wise)
  blocks = nzblocks(T)

  # TODO: Use `permute.(blocks, perm)`.
  blocks_perm = BlockArrays.Block.(permute.(getfield.(blocks, :n), Ref(perm)))

  # Combine the nonzero blocks (dimension-wise)
  blocks_perm_comb = combine_dims(blocks_perm, inds_perm, combdims_perm)

  # Permute the blocks (within the newly combined dimension)
  comb_ind_loc = minimum(combdims_perm)
  blocks_perm_comb = perm_blocks(blocks_perm_comb, comb_ind_loc, blockperm)
  blocks_perm_comb = sort(blocks_perm_comb; lt=isblockless)

  # Combine the blocks (within the newly combined and permuted dimension)
  blocks_perm_comb = combine_blocks(blocks_perm_comb, comb_ind_loc, blockcomb)
  blocktype = set_ndims(unwrap_type(T), length(is))
  return BlockSparseArray{eltype(T),length(is),blocktype}(undef, blocks_perm_comb, is)
end
