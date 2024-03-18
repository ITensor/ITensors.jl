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

function blocksparse_blocks(a::AbstractArray)
  return blocks(a)
end

function blocksparse_getindex(a::AbstractArray{<:Any,N}, I::Vararg{Int,N}) where {N}
  @boundscheck checkbounds(a, I...)
  return a[findblockindex.(axes(a), I)...]
end

# TODO: Implement as `copy(@view a[I...])`, which is then implemented
# through `ArrayLayouts.sub_materialize`.
using ..SparseArrayInterface: set_getindex_zero_function
function blocksparse_getindex(
  a::AbstractArray{<:Any,N}, I::Vararg{AbstractVector{<:Block{1}},N}
) where {N}
  blocks_a = blocks(a)
  # Convert to cartesian indices of the underlying sparse array of blocks.
  CI = map(i -> Int.(i), I)
  subblocks_a = blocks_a[CI...]
  subaxes = ntuple(ndims(a)) do i
    return only(axes(axes(a, i)[I[i]]))
  end
  subblocks_a = set_getindex_zero_function(subblocks_a, BlockZero(subaxes))
  return typeof(a)(subblocks_a, subaxes)
end

# Slice by block and merge according to the blocking structure of the indices.
function blocksparse_getindex(
  a::AbstractArray{<:Any,N}, I::Vararg{AbstractBlockVector{<:Block{1}},N}
) where {N}
  a_sub = a[Vector.(I)...]
  # TODO: Define `blocklengths(::AbstractBlockVector)`? Maybe make a PR
  # to `BlockArrays`.
  blockmergers = blockedrange.(blocklengths.(only.(axes.(I))))
  # TODO: Need to implement this!
  a_merged = block_merge(a_sub, blockmergers...)
  return a_merged
end

# TODO: Need to implement this!
function block_merge(a::AbstractArray{<:Any,N}, I::Vararg{BlockedUnitRange,N}) where {N}
  # Need to `block_merge` each axis.
  return a
end

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
  blocksparse_blocks(a)[i...] = value
  return a
end

function blocksparse_viewblock(a::AbstractArray{<:Any,N}, I::Block{N}) where {N}
  # TODO: Create a conversion function, say `CartesianIndex(Int.(Tuple(I)))`.
  i = I.n
  @boundscheck blockcheckbounds(a, i...)
  return blocksparse_blocks(a)[i...]
end

function block_nstored(a::AbstractArray)
  return nstored(blocksparse_blocks(a))
end

# Base

# PermutedDimsArray
function blocksparse_blocks(a::PermutedDimsArray)
  blocks_parent = blocksparse_blocks(parent(a))
  # Lazily permute each block
  blocks_parent_mapped = mappedarray(
    Base.Fix2(PermutedDimsArray, perm(a)),
    Base.Fix2(PermutedDimsArray, iperm(a)),
    blocks_parent,
  )
  return PermutedDimsArray(blocks_parent_mapped, perm(a))
end
