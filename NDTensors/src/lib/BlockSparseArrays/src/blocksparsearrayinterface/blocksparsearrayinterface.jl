using BlockArrays:
  AbstractBlockVector,
  Block,
  BlockedUnitRange,
  BlockIndex,
  block,
  blockcheckbounds,
  blocks,
  blocklengths,
  findblockindex
using ..SparseArrayInterface: perm, iperm, nstored
using MappedArrays: mappedarray

blocksparse_blocks(a::AbstractArray) = error("Not implemented")

function blocksparse_getindex(a::AbstractArray{<:Any,N}, I::Vararg{Int,N}) where {N}
  @boundscheck checkbounds(a, I...)
  return a[findblockindex.(axes(a), I)...]
end

## # TODO: Implement as `copy(@view a[I...])`, which is then implemented
## # through `ArrayLayouts.sub_materialize`.
## function blocksparse_getindex(
##   a::AbstractArray{<:Any,N}, I::Vararg{AbstractVector{<:Block{1}},N}
## ) where {N}
##   blocks_a = blocks(a)
##   # Convert to cartesian indices of the underlying sparse array of blocks.
##   CI = map(i -> Int.(i), I)
##   subblocks_a = blocks_a[CI...]
##   subaxes = ntuple(ndims(a)) do i
##     return axes(a, i)[I[i]]
##   end
##   return typeof(a)(subblocks_a, subaxes)
## end
##
## # Slice by block and merge according to the blocking structure of the indices.
## function blocksparse_getindex(
##   a::AbstractArray{<:Any,N}, I::Vararg{AbstractBlockVector{<:Block{1}},N}
## ) where {N}
##   a_sub = a[Vector.(I)...]
##   # TODO: Define `blocklengths(::AbstractBlockVector)`? Maybe make a PR
##   # to `BlockArrays`.
##   blockmergers = blockedrange.(blocklengths.(only.(axes.(I))))
##   # TODO: Need to implement this!
##   a_merged = block_merge(a_sub, blockmergers...)
##   return a_merged
## end
##
## # TODO: Need to implement this!
## function block_merge(a::AbstractArray{<:Any,N}, I::Vararg{BlockedUnitRange,N}) where {N}
##   # Need to `block_merge` each axis.
##   return a
## end

# TODO: Need to implement this!
function block_merge end

function blocksparse_setindex!(a::AbstractArray{<:Any,N}, value, I::Vararg{Int,N}) where {N}
  @boundscheck checkbounds(a, I...)
  a[findblockindex.(axes(a), I)...] = value
  return a
end

function blocksparse_setindex!(a::AbstractArray{<:Any,N}, value, I::BlockIndex{N}) where {N}
  a_b = view(a, block(I))
  a_b[I.α...] = value
  # Set the block, required if it is structurally zero
  a[block(I)] = a_b
  return a
end

function blocksparse_setindex!(a::AbstractArray{<:Any,N}, value, I::Block{N}) where {N}
  # TODO: Create a conversion function, say `CartesianIndex(Int.(Tuple(I)))`.
  i = I.n
  @boundscheck blockcheckbounds(a, i...)
  blocks(a)[i...] = value
  return a
end

function blocksparse_viewblock(a::AbstractArray{<:Any,N}, I::Block{N}) where {N}
  # TODO: Create a conversion function, say `CartesianIndex(Int.(Tuple(I)))`.
  i = I.n
  @boundscheck blockcheckbounds(a, i...)
  return blocks(a)[i...]
end

function block_nstored(a::AbstractArray)
  return nstored(blocks(a))
end

# BlockArrays

function blocksparse_blocks(a::PermutedDimsArray)
  blocks_parent = blocks(parent(a))
  # Lazily permute each block
  blocks_parent_mapped = mappedarray(
    Base.Fix2(PermutedDimsArray, perm(a)),
    Base.Fix2(PermutedDimsArray, iperm(a)),
    blocks_parent,
  )
  return PermutedDimsArray(blocks_parent_mapped, perm(a))
end

function blockindices(a::AbstractArray, block::Block, indices::Tuple)
  return blockindices(axes(a), block, indices)
end

function blockindices(axes::Tuple, block::Block, indices::Tuple)
  return blockindices.(axes, Tuple(block), indices)
end

function blockindices(axis::AbstractUnitRange, block::Block, indices)
  indices_within_block = intersect(indices, axis[block])
  if iszero(length(indices_within_block))
    # Falls outside of block
    return 1:0
  end
  return only(blockindexrange(axis, indices_within_block).indices)
end

function blocksparse_blocks(a::SubArray)
  # First slice blockwise.
  blockranges = blockrange.(axes(parent(a)), a.indices)
  # Then slice the blocks.
  sliced_blocks = map(stored_indices(blocks(parent(a)))) do index
    tuple_index = Tuple(index)
    block = Block(tuple_index)
    return view(
      blocks(parent(a))[tuple_index...], blockindices(parent(a), block, a.indices)...
    )
  end
  # TODO: Use a `set_data` function, or some kind of `similar` or `zero` method?
  blocks_a_sub = SparseArrayDOK(
    sliced_blocks, size(blocks(parent(a))), blocks(parent(a)).zero
  )
  # TODO: Avoid copying, use a view?
  return blocks_a_sub[map(blockrange -> Int.(blockrange), blockranges)...]
end
