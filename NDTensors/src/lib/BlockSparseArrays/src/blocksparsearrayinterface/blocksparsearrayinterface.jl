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
## using MappedArrays: mappedarray

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

using ..SparseArrayInterface: SparseArrayInterface, AbstractSparseArray

# Represents the array of arrays of a `SubArray`
# wrapping a block spare array, i.e. `blocks(array)` where `a` is a `SubArray`.
struct SparsePermutedDimsArrayBlocks{T,N,Array<:PermutedDimsArray{T,N}} <:
       AbstractSparseArray{T,N}
  array::Array
end
function blocksparse_blocks(a::PermutedDimsArray)
  return SparsePermutedDimsArrayBlocks(a)
end
_perm(::PermutedDimsArray{<:Any,<:Any,P}) where {P} = P
_getindices(t::Tuple, indices) = map(i -> t[i], indices)
_getindices(i::CartesianIndex, indices) = CartesianIndex(_getindices(Tuple(i), indices))
function SparseArrayInterface.stored_indices(a::SparsePermutedDimsArrayBlocks)
  return map(I -> _getindices(I, _perm(a.array)), stored_indices(blocks(parent(a.array))))
end
function Base.size(a::SparsePermutedDimsArrayBlocks)
  return _getindices(size(blocks(parent(a.array))), _perm(a.array))
end
function Base.getindex(a::SparsePermutedDimsArrayBlocks, index::Vararg{Int})
  return PermutedDimsArray(
    blocks(parent(a.array))[_getindices(index, _perm(a.array))...], _perm(a.array)
  )
end
function SparseArrayInterface.sparse_storage(a::SparsePermutedDimsArrayBlocks)
  return error("Not implemented")
end

# TODO: Move to `BlockArraysExtensions`.
function blockindices(a::AbstractArray, block::Block, indices::Tuple)
  return blockindices(axes(a), block, indices)
end

# TODO: Move to `BlockArraysExtensions`.
function blockindices(axes::Tuple, block::Block, indices::Tuple)
  return blockindices.(axes, Tuple(block), indices)
end

# TODO: Move to `BlockArraysExtensions`.
function blockindices(axis::AbstractUnitRange, block::Block, indices)
  indices_within_block = intersect(indices, axis[block])
  if iszero(length(indices_within_block))
    # Falls outside of block
    return 1:0
  end
  return only(blockindexrange(axis, indices_within_block).indices)
end

# Represents the array of arrays of a `SubArray`
# wrapping a block spare array, i.e. `blocks(array)` where `a` is a `SubArray`.
struct SparseSubArrayBlocks{T,N,Array<:SubArray{T,N}} <: AbstractSparseArray{T,N}
  array::Array
end
# TODO: Define this as `blockrange(a::AbstractArray, indices::Tuple{Vararg{AbstractUnitRange}})`.
function blockrange(a::SparseSubArrayBlocks)
  blockranges = blockrange.(axes(parent(a.array)), a.array.indices)
  return map(blockrange -> Int.(blockrange), blockranges)
end
function Base.axes(a::SparseSubArrayBlocks)
  return Base.OneTo.(length.(blockrange(a)))
end
function Base.size(a::SparseSubArrayBlocks)
  return length.(axes(a))
end
function SparseArrayInterface.stored_indices(a::SparseSubArrayBlocks)
  return stored_indices(view(blocks(parent(a.array)), axes(a)...))
end
function Base.getindex(a::SparseSubArrayBlocks{<:Any,N}, I::CartesianIndex{N}) where {N}
  return a[Tuple(I)...]
end
function Base.getindex(a::SparseSubArrayBlocks{<:Any,N}, I::Vararg{Int,N}) where {N}
  parent_blocks = @view blocks(parent(a.array))[axes(a)...]
  parent_block = parent_blocks[I...]
  # TODO: Define this using `blockrange(a::AbstractArray, indices::Tuple{Vararg{AbstractUnitRange}})`.
  block = Block(ntuple(i -> blockrange(a)[i][I[i]], ndims(a)))
  return @view parent_block[blockindices(parent(a.array), block, a.array.indices)...]
end
function Base.setindex!(a::SparseSubArrayBlocks{<:Any,N}, value, I::Vararg{Int,N}) where {N}
  parent_blocks = view(blocks(parent(a.array)), axes(a)...)
  return parent_blocks[I...][blockindices(parent(a.array), Block(I), a.array.indices)...] =
    value
end
function Base.isassigned(a::SparseSubArrayBlocks{<:Any,N}, I::Vararg{Int,N}) where {N}
  # TODO: Implement this properly.
  return true
end
function SparseArrayInterface.sparse_storage(a::SparseSubArrayBlocks)
  return error("Not implemented")
end

function blocksparse_blocks(a::SubArray)
  return SparseSubArrayBlocks(a)
end
