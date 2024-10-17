# Needed for implementing block sparse combiner contraction.
using .BlockSparseArrays: blocks, nonzero_keys
using .BlockSparseArrays.BlockArrays: BlockArrays
# TODO: Move to `BlockSparseArrays`, come up with better name.
# `nonzero_block_keys`?
nzblocks(a::BlockSparseArray) = BlockArrays.Block.(Tuple.(collect(nonzero_keys(blocks(a)))))

# ⊗
# TODO: Rename this function.
function outer(i1::BlockArrays.BlockedUnitRange, i2::BlockArrays.BlockedUnitRange)
  axes = (i1, i2)
  return BlockArrays.blockedrange(
    prod.(length, vec(collect(Iterators.product(BlockArrays.blocks.(axes)...))))
  )
end

outer(i::BlockArrays.BlockedUnitRange) = i

function combine_dims(
  blocks::Vector{BlockArrays.Block{N,Int}}, inds, combdims::NTuple{NC,Int}
) where {N,NC}
  nblcks = map(i -> BlockArrays.blocklength(inds[i]), combdims)
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
  nblcks = map(i -> BlockArrays.blocklength(inds[i]), combdims)
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

# Uncombining utils

# Uncombine the blocks along the dimension dim
# according to the pattern in blockcomb (for example, blockcomb
# is [1,2,2,3] and dim = 2, so the blocks (1,2),(2,3) get
# split into (1,2),(1,3),(2,4))
function uncombine_blocks(
  blocks::Vector{BlockArrays.Block{N,Int}}, dim::Int, blockcomb::Vector{Int}
) where {N}
  blocks_uncomb = Vector{BlockArrays.Block{N,Int}}()
  ncomb_tot = 0
  for i in 1:length(blocks)
    block = blocks[i]
    blockval = block.n[dim]
    ncomb = _number_uncombined(blockval, blockcomb)
    ncomb_shift = _number_uncombined_shift(blockval, blockcomb)
    push!(blocks_uncomb, setindex(block, blockval + ncomb_shift, dim))
    for j in 1:(ncomb-1)
      push!(blocks_uncomb, setindex(block, blockval + ncomb_shift + j, dim))
    end
  end
  return blocks_uncomb
end

function uncombine_block(
  block::BlockArrays.Block{N}, dim::Int, blockcomb::Vector{Int}
) where {N}
  blocks_uncomb = Vector{BlockArrays.Block{N,Int}}()
  ncomb_tot = 0
  blockval = block.n[dim]
  ncomb = _number_uncombined(blockval, blockcomb)
  ncomb_shift = _number_uncombined_shift(blockval, blockcomb)
  push!(blocks_uncomb, setindex(block, blockval + ncomb_shift, dim))
  for j in 1:(ncomb-1)
    push!(blocks_uncomb, setindex(block, blockval + ncomb_shift + j, dim))
  end
  return blocks_uncomb
end

# TODO: Rethink this function.
function reshape(blockT::BlockArrays.Block{NT}, indsT, indsR) where {NT}
  nblocksT = BlockArrays.blocklength.(indsT)
  nblocksR = BlockArrays.blocklength.(indsR)
  blockR = Tuple(
    CartesianIndices(nblocksR)[LinearIndices(nblocksT)[CartesianIndex(blockT.n)]]
  )
  return BlockArrays.Block(blockR)
end
