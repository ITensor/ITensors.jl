using BlockArrays:
  Block, BlockedUnitRange, block, blockindex, blocks, blocksize, findblock, findblockindex
using ..SparseArrayInterface: stored_indices

function blockdiagonal(f!, elt::Type, axes::Tuple)
  a = BlockSparseArray{elt}(axes)
  for i in 1:minimum(blocksize(a))
    b = Block(ntuple(Returns(i), ndims(a)))
    a[b] = f!(a[b])
  end
  return a
end

function cartesianindices(axes::Tuple, b::Block)
  return CartesianIndices(ntuple(dim -> axes[dim][Tuple(b)[dim]], length(axes)))
end

function blockindexrange(axis::BlockedUnitRange, r::UnitRange)
  bi1 = findblockindex(axis, first(r))
  bi2 = findblockindex(axis, last(r))
  b = block(bi1)
  # Range must fall within a single block.
  @assert b == block(bi2)
  i1 = blockindex(bi1)
  i2 = blockindex(bi2)
  return b[i1:i2]
end

function blockindexrange(axes::Tuple, I::CartesianIndices)
  brs = blockindexrange.(axes, I.indices)
  b = Block(block.(brs))
  rs = map(br -> only(br.indices), brs)
  return b[rs...]
end

function blockindexrange(a::AbstractArray, I::CartesianIndices)
  return blockindexrange(axes(a), I)
end

function cartesianindices(a::AbstractArray, b::Block)
  return cartesianindices(axes(a), b)
end

# Output which blocks of `axis` are contained within the unit range `range`.
# The start and end points must match.
function findblocks(axis::AbstractUnitRange, range::AbstractUnitRange)
  # TODO: Add a test that the start and end points of the ranges match.
  return findblock(axis, first(range)):findblock(axis, last(range))
end

function block_stored_indices(a::AbstractArray)
  return Block.(Tuple.(stored_indices(blocks(a))))
end

##############################################################
using BlockArrays: BlockArrays, BlockRange

function map_mismatched_blocking!(f, a_dest::AbstractArray, a_src::AbstractArray)
  # Create a common set of axes with a blocking that includes the
  # blocking of `a_dest` and `a_src`.
  matching_axes = BlockArrays.combine_blockaxes.(axes(a_dest), axes(a_src))
  for b in block_stored_indices(a_src)
    # Get the subblocks of the matching axes
    # TODO: `union` all `subblocks` of all `a_src` and `a_dest`.
    subblocks = BlockRange(
      ntuple(ndims(a_dest)) do dim
        findblocks(matching_axes[dim], axes(a_src, dim)[Tuple(b)[dim]])
      end,
    )
    for subblock in subblocks
      I = cartesianindices(matching_axes, subblock)
      I_dest = blockindexrange(a_dest, I)
      I_src = blockindexrange(a_src, I)

      # TODO: Broken, need to fix.
      # map!(f, view(a_dest, I_dest), view(a_src, I_src))

      map!(f, view(a_dest, I_dest), a_src[I_src])
    end
  end
  return a_dest
end
