#
# BlockOffsets
#

const Blocks{N} = Vector{Block{N}}
const BlockOffset{N} = Pair{Block{N},Int}
# Use Dictionary from Dictionaries.jl (faster
# iteration than Base.Dict)
const BlockOffsets{N} = Dictionary{Block{N},Int}

BlockOffset(block::Block{N}, offset::Int) where {N} = BlockOffset{N}(block, offset)

ndims(::Blocks{N}) where {N} = N
ndims(::BlockOffset{N}) where {N} = N
ndims(::BlockOffsets{N}) where {N} = N

blocktype(bofs::BlockOffsets) = keytype(bofs)

nzblock(bof::BlockOffset) = first(bof)

offset(bof::BlockOffset) = last(bof)

nzblock(block::Block) = block

# Get the offset if the nth block in the block-offsets
# list
offset(bofs::BlockOffsets, n) = offset(bofs[n])

nnzblocks(bofs::BlockOffsets) = length(bofs)
nnzblocks(bs::Blocks) = length(bs)

eachnzblock(bofs::BlockOffsets) = keys(bofs)

nzblocks(bofs::BlockOffsets) = collect(eachnzblock(bofs))

# define block ordering with reverse lexographical order
function isblockless(b1::Block{N}, b2::Block{N}) where {N}
  return CartesianIndex(b1) < CartesianIndex(b2)
end

function isblockless(bof1::BlockOffset{N}, bof2::BlockOffset{N}) where {N}
  return isblockless(nzblock(bof1), nzblock(bof2))
end

function isblockless(bof1::BlockOffset{N}, b2::Block{N}) where {N}
  return isblockless(nzblock(bof1), b2)
end

function isblockless(b1::Block{N}, bof2::BlockOffset{N}) where {N}
  return isblockless(b1, nzblock(bof2))
end

function offset(bofs::BlockOffsets{N}, block::Block{N}) where {N}
  if !isassigned(bofs, block)
    return nothing
  end
  return bofs[block]
end

function nnz(bofs::BlockOffsets, inds)
  _nnz = 0
  nnzblocks(bofs) == 0 && return _nnz
  for block in eachnzblock(bofs)
    _nnz += blockdim(inds, block)
  end
  return _nnz
end

blockoffsets(blocks::Vector{<:NTuple}, inds) = blockoffsets(Block.(blocks), inds)

# TODO: should this be a constructor?
function blockoffsets(blocks::Vector{<:Block{N}}, inds) where {N}
  blockoffsets = BlockOffsets{N}()
  nnz = 0
  for block in blocks
    insert!(blockoffsets, block, nnz)
    current_block_dim = blockdim(inds, block)
    nnz += current_block_dim
  end
  return blockoffsets, nnz
end

"""
    diagblockoffsets(blocks::Blocks,inds)

Get the blockoffsets only along the diagonal.
The offsets are along the diagonal.

Assumes the blocks are allong the diagonal.
"""
function diagblockoffsets(
  blocks::Vector{BlockT}, inds
) where {BlockT<:Union{Block{N},Tuple{Vararg{Any,N}}}} where {N}
  blockoffsets = BlockOffsets{N}()
  nnzdiag = 0
  for (i, block) in enumerate(blocks)
    insert!(blockoffsets, Block(block), nnzdiag)
    current_block_diaglength = blockdiaglength(inds, block)
    nnzdiag += current_block_diaglength
  end
  return blockoffsets, nnzdiag
end

# Permute the blockoffsets and indices
function permutedims(boffs::BlockOffsets{N}, inds, perm::NTuple{N,Int}) where {N}
  blocksR = Blocks{N}(undef, nnzblocks(boffs))
  for (i, block) in enumerate(keys(boffs))
    blocksR[i] = permute(block, perm)
  end
  indsR = permute(inds, perm)
  blockoffsetsR, _ = blockoffsets(blocksR, indsR)
  return blockoffsetsR, indsR
end

function permutedims(blocks::Vector{Block{N}}, perm::NTuple{N,Int}) where {N}
  blocks_perm = Vector{Block{N}}(undef, length(blocks))
  for (i, block) in enumerate(blocks)
    blocks_perm[i] = permute(block, perm)
  end
  return blocks_perm
end

"""
blockdim(T::BlockOffsets,nnz::Int,pos::Int)

Get the block dimension of the block at position pos.
"""
function blockdim(boffs::BlockOffsets, nnz::Int, pos::Int)
  if nnzblocks(boffs) == 0
    return 0
  elseif pos == nnzblocks(boffs)
    return nnz - offset(boffs, pos)
  end
  return offset(boffs, pos + 1) - offset(boffs, pos)
end

function Base.union(
  boffs1::BlockOffsets{N}, nnz1::Int, boffs2::BlockOffsets{N}, nnz2::Int
) where {N}
  n1, n2 = 1, 1
  boffsR = BlockOffset{N}[]
  current_offset = 0
  while n1 <= length(boffs1) && n2 <= length(boffs2)
    if isblockless(boffs1[n1], boffs2[n2])
      push!(boffsR, BlockOffset(nzblock(boffs1[n1]), current_offset))
      current_offset += blockdim(boffs1, nnz1, n1)
      n1 += 1
    elseif isblockless(boffs2[n2], boffs1[n1])
      push!(boffsR, BlockOffset(nzblock(boffs2[n2]), current_offset))
      current_offset += blockdim(boffs2, nnz2, n2)
      n2 += 1
    else
      push!(boffsR, BlockOffset(nzblock(boffs1[n1]), current_offset))
      current_offset += blockdim(boffs1, nnz1, n1)
      n1 += 1
      n2 += 1
    end
  end
  if n1 <= length(boffs1)
    for n in n1:length(boffs1)
      push!(boffsR, BlockOffset(nzblock(boffs1[n]), current_offset))
      current_offset += blockdim(boffs1, nnz1, n)
    end
  elseif n2 <= length(boffs2)
    for n in n2:length(bofss2)
      push!(boffsR, BlockOffset(nzblock(boffs2[n]), current_offset))
      current_offset += blockdim(boffs2, nnz2, n)
    end
  end
  return boffsR, current_offset
end
