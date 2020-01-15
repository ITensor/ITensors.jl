export BlockSparse,
       BlockSparseTensor,
       Block,
       block,
       BlockOffset,
       BlockOffsets,
       blockoffsets,
       blockview,
       nnzblocks,
       nnz,
       findblock,
       isblocknz

#
# BlockOffsets
#

const Block{N} = NTuple{N,Int}
const BlockOffset{N} = Pair{Block{N},Int}
const BlockOffsets{N} = Vector{BlockOffset{N}}

BlockOffset(block::Block{N},offset::Int) where {N} = BlockOffset{N}(block,offset)

block(bof::BlockOffset) = first(bof)
offset(bof::BlockOffset) = last(bof)
block(block::Block) = block

nnzblocks(bofs::BlockOffsets) = length(bofs)

# define block ordering with reverse lexographical order
function isblockless(b1::Block{N},
                     b2::Block{N}) where {N}
  return CartesianIndex(b1) < CartesianIndex(b2)
end

function isblockless(bof1::BlockOffset{N},
                     bof2::BlockOffset{N}) where {N}
  return isblockless(block(bof1),block(bof2))
end

function isblockless(bof1::BlockOffset{N},
                     b2::Block{N}) where {N}
  return isblockless(block(bof1),b2)
end

function isblockless(b1::Block{N},
                     bof2::BlockOffset{N}) where {N}
  return isblockless(b1,block(bof2))
end

function check_blocks_sorted(blockoffsets::BlockOffsets)
  for jj in 1:length(blockoffsets)-1
    block_jj = block(blockoffsets[jj])
    block_jj1 = block(blockoffsets[jj+1])
    if !isblockless(block_jj,block_jj1)
      error("Blocks in BlockOffsets not ordered")
    end
  end
  return
end

# Get the offset if the nth block in the block-offsets
# list
offset(bofs::BlockOffsets,n::Int) = offset(bofs[n])

function offset(bofs::BlockOffsets{N},
                block::Block{N}) where {N}
  block_pos = findblock(bofs,block)
  isnothing(block_pos) && return nothing
  return offset(bofs,block_pos)
end

"""
findblock(::BlockOffsets,::Block)

Output the index of the specified block in the block-offsets
list.
If not found, return nothing.
Searches assuming the blocks are sorted.
If more than one block exists, throw an error.
"""
function findblock(bofs::BlockOffsets{N},
                   block::Block{N}) where {N}
  r = searchsorted(bofs,block;lt=isblockless)
  length(r)>1 && error("In findblock, more than one block found")   
  length(r)==0 && return nothing
  return first(r)
end

function nnz(bofs::BlockOffsets,inds)
  nnzblocks(bofs) == 0 && return 0
  lastblock,lastoffset = bofs[end]
  return lastoffset + blockdim(inds,lastblock)
end

"""
new_block_pos(::BlockOffsets,::Block)

Output the index where the specified block should go in
the block-offsets list.
Searches assuming the blocks are sorted.
If the block already exists, throw an error.
"""
function new_block_pos(bofs::BlockOffsets{N},
                       block::Block{N}) where {N}
  r = searchsorted(bofs,block;lt=isblockless)
  length(r)>1 && error("In new_block_pos, more than one block found")
  length(r)==1 && error("In new_block_pos, block already found")
  return first(r)
end

# TODO: should this be a constructor?
function get_blockoffsets(blocks::Vector{Block{N}},
                          inds) where {N}
  blocks = sort(blocks;lt=isblockless)
  blockoffsets = BlockOffsets{N}(undef,length(blocks))
  offset_total = 0
  for (i,block) in enumerate(blocks)
    blockoffsets[i] = block=>offset_total
    current_block_dim = blockdim(inds,block)
    offset_total += current_block_dim
  end
  return blockoffsets,offset_total
end

# Permute the blockoffsets and indices
function permute(blockoffsets::BlockOffsets{N},
                 inds,
                 perm::NTuple{N,Int}) where {N}
  blocksR = Vector{Block{N}}(undef,nnzblocks(blockoffsets))
  for (i,(block,offset)) in enumerate(blockoffsets)
    blocksR[i] = permute(block,perm)
  end
  indsR = permute(inds,perm)
  blockoffsetsR,_ = get_blockoffsets(blocksR,indsR)
  return blockoffsetsR,indsR
end

"""
blockdim(T::BlockOffsets,nnz::Int,pos::Int)

Get the block dimension of the block at position pos.
"""
function blockdim(boffs::BlockOffsets,
                  nnz::Int,
                  pos::Int)
  if nnzblocks(boffs)==0
    return 0
  elseif pos==nnzblocks(boffs)
    return nnz-offset(boffs,pos)
  end
  return offset(boffs,pos+1)-offset(boffs,pos)
end

function Base.union(boffs1::BlockOffsets{N},
                    nnz1::Int,
                    boffs2::BlockOffsets{N},
                    nnz2::Int) where {N}
  n1,n2 = 1,1
  boffsR = BlockOffset{N}[]
  current_offset = 0
  while n1 <= length(boffs1) && n2 <= length(boffs2)
    if isblockless(boffs1[n1],boffs2[n2])
      push!(boffsR, BlockOffset(block(boffs1[n1]),current_offset))
      current_offset += blockdim(boffs1,nnz1,n1)
      n1 += 1
    elseif isblockless(boffs2[n2],boffs1[n1])
      push!(boffsR, BlockOffset(block(boffs2[n2]),current_offset))
      current_offset += blockdim(boffs2,nnz2,n2)
      n2 += 1
    else
      push!(boffsR, BlockOffset(block(boffs1[n1]),current_offset))
      current_offset += blockdim(boffs1,nnz1,n1)
      n1 += 1
      n2 += 1
    end
  end
  if n1 <= length(boffs1)
    for n in n1:length(boffs1)
      push!(boffsR, BlockOffset(block(boffs1[n]),current_offset))
      current_offset += blockdim(boffs1,nnz1,n)
    end
  elseif n2 <= length(boffs2)
    for n in n2:length(bofss2)
      push!(boffsR, BlockOffset(block(boffs2[n]),current_offset))
      current_offset += blockdim(boffs2,nnz2,n)
    end
  end
  return boffsR,current_offset
end

