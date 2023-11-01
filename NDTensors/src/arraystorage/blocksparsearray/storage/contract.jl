parenttype(::Type{<:BlockSparseArray{<:Any,<:Any,P}}) where {P} = P

permutedims(a::BlockSparseArray, perm) = Base.permutedims(a, perm)

function contract!!(
  ::ArrayStorageTensor,
  ::Any,
  tensor::Tensor{T,N,<:BlockSparseArray{T,N}},
  tensor_labels,
  combiner_tensor::CombinerTensor,
  combiner_tensor_labels,
) where {T,N}
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

function contract!!(
  a::ArrayStorageTensor,
  b::Any,
  combiner_tensor::CombinerTensor,
  combiner_tensor_labels,
  tensor::Tensor{T,N,<:BlockSparseArray{T,N}},
  tensor_labels,
) where {T,N}
  return contract!!(a, b, tensor, tensor_labels, combiner_tensor, combiner_tensor_labels)
end

function permutedims_combine(
  T::Tensor{ElT,N,<:BlockSparseArray{ElT,N}},
  is,
  perm::NTuple{N,Int},
  combdims::NTuple{NC,Int},
  blockperm::Vector{Int},
  blockcomb::Vector{Int},
) where {ElT,N,NC}
  R = permutedims_combine_output(T, is, perm, combdims, blockperm, blockcomb)

  # Permute the indices
  inds_perm = permute(inds(T), perm)

  # Now that the indices are permuted, compute
  # which indices are now combined
  combdims_perm = sort(_permute_combdims(combdims, perm))
  comb_ind_loc = minimum(combdims_perm)

  # Determine the new index before combining
  inds_to_combine = getindices(inds_perm, combdims_perm)
  ind_comb = ⊗(inds_to_combine...)
  ind_comb = permuteblocks(ind_comb, blockperm)

  for b in nzblocks(storage(T))
    Tb = @view T[b]
    b_perm = permute(b, perm)
    b_perm_comb = combine_dims(b_perm, inds_perm, combdims_perm)
    b_perm_comb = perm_block(b_perm_comb, comb_ind_loc, blockperm)
    b_in_combined_dim = b_perm_comb[comb_ind_loc]
    new_b_in_combined_dim = blockcomb[b_in_combined_dim]
    offset = 0
    pos_in_new_combined_block = 1
    while b_in_combined_dim - pos_in_new_combined_block > 0 &&
      blockcomb[b_in_combined_dim - pos_in_new_combined_block] == new_b_in_combined_dim
      offset += blockdim(ind_comb, b_in_combined_dim - pos_in_new_combined_block)
      pos_in_new_combined_block += 1
    end
    b_new = setindex(b_perm_comb, new_b_in_combined_dim, comb_ind_loc)

    Rb_total = blockview(R, b_new)
    dimsRb_tot = dims(Rb_total)
    subind = ntuple(
      i -> if i == comb_ind_loc
        range(1 + offset; stop=offset + blockdim(ind_comb, b_in_combined_dim))
      else
        range(1; stop=dimsRb_tot[i])
      end,
      N - NC + 1,
    )
    Rb = @view array(Rb_total)[subind...]

    # XXX Are these equivalent?
    #Tb_perm = permutedims(Tb,perm)
    #copyto!(Rb,Tb_perm)

    # XXX Not sure what this was for
    Rb = reshape(Rb, permute(dims(Tb), perm))
    # TODO: Make this `convert` call more general
    # for GPUs.
    Tbₐ = convert(Array, Tb)
    ## @strided Rb .= permutedims(Tbₐ, perm)
    permutedims!(Rb, Tbₐ, perm)
  end

  return R
end

function permutedims_combine_output(
  T::Tensor{ElT,N,<:BlockSparseArray{ElT,N}},
  is,
  perm::NTuple{N,Int},
  combdims::NTuple{NC,Int},
  blockperm::Vector{Int},
  blockcomb::Vector{Int},
) where {ElT,N,NC}
  # Permute the indices
  indsT = inds(T)
  inds_perm = permute(indsT, perm)

  # Now that the indices are permuted, compute
  # which indices are now combined
  combdims_perm = sort(_permute_combdims(combdims, perm))

  # Permute the nonzero blocks (dimension-wise)
  blocks = nzblocks(T)

  @show blocks
  @show perm

  # TODO: Use `permute.(blocks, perm)`.
  blocks_perm = permute.(blocks, Ref(perm))

  @show blocks_perm

  # Combine the nonzero blocks (dimension-wise)
  blocks_perm_comb = combine_dims(blocks_perm, inds_perm, combdims_perm)

  # Permute the blocks (within the newly combined dimension)
  comb_ind_loc = minimum(combdims_perm)
  blocks_perm_comb = perm_blocks(blocks_perm_comb, comb_ind_loc, blockperm)
  blocks_perm_comb = sort(blocks_perm_comb; lt=isblockless)

  # Combine the blocks (within the newly combined and permuted dimension)
  blocks_perm_comb = combine_blocks(blocks_perm_comb, comb_ind_loc, blockcomb)

  ## return BlockSparseTensor(leaf_parenttype(T), blocks_perm_comb, is)
  blockinds = map(i -> [blockdim(i, b) for b in 1:nblocks(i)], is)
  blocktype = set_ndims(leaf_parenttype(T), ndims(T))
  return tensor(
    BlockSparseArray{eltype(T),ndims(T),blocktype}(undef, blocks_perm_comb, blockinds), is
  )
end

function combine_dims(
  blocks::Dictionary{CartesianIndex{N},BlockArrays.Block{N,Int}},
  inds,
  combdims::NTuple{NC,Int},
) where {N,NC}
  nblcks = nblocks(inds, combdims)
  blocks_comb = Vector{BlockArrays.Block{N - NC + 1,Int}}(undef, length(blocks))
  for (i, block) in enumerate(blocks)
    blocks_comb[i] = combine_dims(block, inds, combdims)
  end
  return blocks_comb
end

# In the dimension dim, permute the blocks
function perm_blocks(blocks::Vector{BlockArrays.Block{N,Int}}, dim::Int, perm) where {N}
  blocks_perm = Vector{BlockArrays.Block{N,Int}}(undef, length(blocks))
  iperm = invperm(perm)
  for (i, block) in enumerate(blocks)
    blocks_perm[i] = setindex(block, iperm[block[dim]], dim)
  end
  return blocks_perm
end

# In the dimension dim, combine the specified blocks
function combine_blocks(
  blocks::Vector{<:BlockArrays.Block}, dim::Int, blockcomb::Vector{Int}
)
  blocks_comb = copy(blocks)
  nnz_comb = length(blocks)
  for (i, block) in enumerate(blocks)
    dimval = block[dim]
    blocks_comb[i] = setindex(block, blockcomb[dimval], dim)
  end
  unique!(blocks_comb)
  return blocks_comb
end

# Needed for implementing block sparse combiner contraction.
using .BlockSparseArrays: blocks, nonzero_keys
using .BlockSparseArrays.BlockArrays: BlockArrays
# TODO: Move to `BlockSparseArrays`, come up with better name.
# `nonzero_block_keys`?
nzblocks(a::BlockSparseArray) = BlockArrays.Block.(Tuple.(nonzero_keys(blocks(a))))

## # TODO: Implement.
## function contraction_output(tensor1::BlockSparseArray, tensor2::BlockSparseArray, indsR)
##   return error("Not implemented")
## end
## 
## # TODO: Implement.
## function contract!(
##   tensorR::BlockSparseArray,
##   labelsR,
##   tensor1::BlockSparseArray,
##   labels1,
##   tensor2::BlockSparseArray,
##   labels2,
## )
##   return error("Not implemented")
## end
