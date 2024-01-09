using BlockArrays: Block, BlockIndex, block, blocks, blockcheckbounds, findblockindex
using ..SparseArrayInterface: perm, iperm, nstored
using MappedArrays: mappedarray

function blocksparse_blocks(a::AbstractArray)
  return blocks(a)
end

function blocksparse_getindex(a::AbstractArray{<:Any,N}, I::Vararg{Int,N}) where {N}
  @boundscheck checkbounds(a, I...)
  return a[findblockindex.(axes(a), I)...]
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
