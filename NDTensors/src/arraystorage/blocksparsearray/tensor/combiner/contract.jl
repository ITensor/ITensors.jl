# Needed for implementing block sparse combiner contraction.
using .BlockSparseArrays: blocks, nonzero_keys
using .BlockSparseArrays.BlockArrays: BlockArrays
# TODO: Move to `BlockSparseArrays`, come up with better name.
# `nonzero_block_keys`?
nzblocks(a::BlockSparseArray) = BlockArrays.Block.(Tuple.(collect(nonzero_keys(blocks(a)))))

function contract!!(
  tensor_dest::ArrayStorageTensor,
  tensor_dest_labels::Any,
  tensor::Tensor{T,N,<:BlockSparseArray{T,N}},
  tensor_labels,
  combiner_tensor::CombinerTensor,
  combiner_tensor_labels,
) where {T,N}
  is_combining_contraction = is_combining(
    tensor, tensor_labels, combiner_tensor, combiner_tensor_labels
  )
  if is_combining_contraction
    return contract_combine!!(tensor_dest, tensor_dest_labels, tensor, tensor_labels, combiner_tensor, combiner_tensor_labels)
  else # Uncombining
    return contract_uncombine!!(tensor_dest, tensor_dest_labels, tensor, tensor_labels, combiner_tensor, combiner_tensor_labels)
  end
  return invalid_combiner_contraction_error(
    combiner_tensor, tensor_labels, tensor, tensor_labels
  )
end

function contract!!(
  tensor_dest::ArrayStorageTensor,
  tensor_dest_labels::Any,
  combiner_tensor::CombinerTensor,
  combiner_tensor_labels,
  tensor::Tensor{T,N,<:BlockSparseArray{T,N}},
  tensor_labels,
) where {T,N}
  return contract!!(tensor_dest, tensor_dest_labels, tensor, tensor_labels, combiner_tensor, combiner_tensor_labels)
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
    Tb = @view storage(T)[b]
    b_perm = permute(b, perm)
    b_perm_comb = combine_dims(b_perm, inds_perm, combdims_perm)
    b_perm_comb = perm_block(b_perm_comb, comb_ind_loc, blockperm)
    b_in_combined_dim = b_perm_comb.n[comb_ind_loc]
    new_b_in_combined_dim = blockcomb[b_in_combined_dim]
    offset = 0
    pos_in_new_combined_block = 1
    while b_in_combined_dim - pos_in_new_combined_block > 0 &&
      blockcomb[b_in_combined_dim - pos_in_new_combined_block] == new_b_in_combined_dim
      offset += blockdim(ind_comb, b_in_combined_dim - pos_in_new_combined_block)
      pos_in_new_combined_block += 1
    end
    b_new = setindex(b_perm_comb, new_b_in_combined_dim, comb_ind_loc)

    # TODO: Define block view for Tensor?
    Rb_total = @view storage(R)[b_new]
    dimsRb_tot = size(Rb_total)
    subind = ntuple(
      i -> if i == comb_ind_loc
        range(1 + offset; stop=offset + blockdim(ind_comb, b_in_combined_dim))
      else
        range(1; stop=dimsRb_tot[i])
      end,
      N - NC + 1,
    )

    Rb = @view Rb_total[subind...]

    # XXX Are these equivalent?
    #Tb_perm = permutedims(Tb,perm)
    #copyto!(Rb,Tb_perm)

    # XXX Not sure what this was for
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

  ## return BlockSparseTensor(unwrap_type(T), blocks_perm_comb, is)
  blockinds = map(i -> [blockdim(i, b) for b in 1:nblocks(i)], is)
  blocktype = set_ndims(unwrap_type(T), length(is))
  return tensor(
    BlockSparseArray{eltype(T),length(is),blocktype}(undef, blocks_perm_comb, blockinds), is
  )
end

function combine_dims(
  blocks::Vector{BlockArrays.Block{N,Int}}, inds, combdims::NTuple{NC,Int}
) where {N,NC}
  nblcks = nblocks(inds, combdims)
  blocks_comb = Vector{BlockArrays.Block{N - NC + 1,Int}}(undef, length(blocks))
  for (i, block) in enumerate(blocks)
    blocks_comb[i] = combine_dims(block, inds, combdims)
  end
  return blocks_comb
end

function getindices(b::BlockArrays.Block, I::Tuple)
  return getindices(b.n, I)
end
deleteat(b::BlockArrays.Block, pos) = BlockArrays.Block(deleteat(b.n, pos))
function insertafter(b::BlockArrays.Block, val, pos)
  return BlockArrays.Block(insertafter(b.n, Int.(val), pos))
end
setindex(b::BlockArrays.Block, val, pos) = BlockArrays.Block(setindex(b.n, Int(val), pos))
permute(s::BlockArrays.Block, perm::Tuple) = BlockArrays.Block(permute(s.n, perm))
# define block ordering with reverse lexographical order
function isblockless(b1::BlockArrays.Block{N}, b2::BlockArrays.Block{N}) where {N}
  return CartesianIndex(b1.n) < CartesianIndex(b2.n)
end
# In the dimension dim, permute the block
function perm_block(block::BlockArrays.Block, dim::Int, perm)
  iperm = invperm(perm)
  return setindex(block, iperm[block.n[dim]], dim)
end

function combine_dims(block::BlockArrays.Block, inds, combdims::NTuple{NC,Int}) where {NC}
  nblcks = nblocks(inds, combdims)
  slice = getindices(block, combdims)
  slice_comb = LinearIndices(nblcks)[slice...]
  block_comb = deleteat(block, combdims)
  block_comb = insertafter(block_comb, tuple(slice_comb), minimum(combdims) - 1)
  return block_comb
end

# In the dimension dim, permute the blocks
function perm_blocks(blocks::Vector{BlockArrays.Block{N,Int}}, dim::Int, perm) where {N}
  blocks_perm = Vector{BlockArrays.Block{N,Int}}(undef, length(blocks))
  iperm = invperm(perm)
  for (i, block) in enumerate(blocks)
    blocks_perm[i] = setindex(block, iperm[block.n[dim]], dim)
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
    dimval = block.n[dim]
    blocks_comb[i] = setindex(block, blockcomb[dimval], dim)
  end
  unique!(blocks_comb)
  return blocks_comb
end
