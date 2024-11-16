using BlockArrays:
  AbstractBlockVector,
  Block,
  BlockIndex,
  BlockRange,
  BlockSlice,
  BlockVector,
  BlockedUnitRange,
  BlockedVector,
  block,
  blockcheckbounds,
  blocklengths,
  blocks,
  findblockindex
using LinearAlgebra: Adjoint, Transpose
using ..SparseArraysBase: perm, iperm, stored_length, sparse_zero!

blocksparse_blocks(a::AbstractArray) = error("Not implemented")

blockstype(a::AbstractArray) = blockstype(typeof(a))

function blocksparse_getindex(a::AbstractArray{<:Any,N}, I::Vararg{Int,N}) where {N}
  @boundscheck checkbounds(a, I...)
  return a[findblockindex.(axes(a), I)...]
end

# Fix ambiguity error.
function blocksparse_getindex(a::AbstractArray{<:Any,0})
  # TODO: Use `Block()[]` once https://github.com/JuliaArrays/BlockArrays.jl/issues/430
  # is fixed.
  return a[BlockIndex{0,Tuple{},Tuple{}}((), ())]
end

# a[1:2, 1:2]
# TODO: This definition means that the result of slicing a block sparse array
# with a non-blocked unit range is blocked. We may want to change that behavior,
# and make that explicit with `@blocked a[1:2, 1:2]`. See the discussion in
# https://github.com/JuliaArrays/BlockArrays.jl/issues/347 and also
# https://github.com/ITensor/ITensors.jl/issues/1336.
function blocksparse_to_indices(a, inds, I::Tuple{UnitRange{<:Integer},Vararg{Any}})
  bs1 = to_blockindices(inds[1], I[1])
  I1 = BlockSlice(bs1, blockedunitrange_getindices(inds[1], I[1]))
  return (I1, to_indices(a, Base.tail(inds), Base.tail(I))...)
end

# Special case when there is no blocking.
function blocksparse_to_indices(
  a,
  inds::Tuple{Base.OneTo{<:Integer},Vararg{Any}},
  I::Tuple{UnitRange{<:Integer},Vararg{Any}},
)
  return (inds[1][I[1]], to_indices(a, Base.tail(inds), Base.tail(I))...)
end

# a[[Block(2), Block(1)], [Block(2), Block(1)]]
function blocksparse_to_indices(a, inds, I::Tuple{Vector{<:Block{1}},Vararg{Any}})
  I1 = BlockIndices(I[1], blockedunitrange_getindices(inds[1], I[1]))
  return (I1, to_indices(a, Base.tail(inds), Base.tail(I))...)
end

# a[mortar([Block(1)[1:2], Block(2)[1:3]]), mortar([Block(1)[1:2], Block(2)[1:3]])]
# a[[Block(1)[1:2], Block(2)[1:3]], [Block(1)[1:2], Block(2)[1:3]]]
function blocksparse_to_indices(
  a, inds, I::Tuple{BlockVector{<:BlockIndex{1},<:Vector{<:BlockIndexRange{1}}},Vararg{Any}}
)
  I1 = BlockIndices(I[1], blockedunitrange_getindices(inds[1], I[1]))
  return (I1, to_indices(a, Base.tail(inds), Base.tail(I))...)
end

# a[BlockVector([Block(2), Block(1)], [2]), BlockVector([Block(2), Block(1)], [2])]
# Permute and merge blocks.
# TODO: This isn't merging blocks yet, that needs to be implemented that.
function blocksparse_to_indices(
  a, inds, I::Tuple{AbstractBlockVector{<:Block{1}},Vararg{Any}}
)
  I1 = BlockIndices(I[1], blockedunitrange_getindices(inds[1], I[1]))
  return (I1, to_indices(a, Base.tail(inds), Base.tail(I))...)
end

# TODO: Need to implement this!
function block_merge end

function blocksparse_setindex!(a::AbstractArray{<:Any,N}, value, I::Vararg{Int,N}) where {N}
  @boundscheck checkbounds(a, I...)
  a[findblockindex.(axes(a), I)...] = value
  return a
end

# Fix ambiguity error.
function blocksparse_setindex!(a::AbstractArray{<:Any,0}, value)
  # TODO: Use `Block()[]` once https://github.com/JuliaArrays/BlockArrays.jl/issues/430
  # is fixed.
  a[BlockIndex{0,Tuple{},Tuple{}}((), ())] = value
  return a
end

function blocksparse_setindex!(a::AbstractArray{<:Any,N}, value, I::BlockIndex{N}) where {N}
  i = Int.(Tuple(block(I)))
  a_b = blocks(a)[i...]
  a_b[I.α...] = value
  # Set the block, required if it is structurally zero.
  blocks(a)[i...] = a_b
  return a
end

# Fix ambiguity error.
function blocksparse_setindex!(a::AbstractArray{<:Any,0}, value, I::BlockIndex{0})
  a_b = blocks(a)[]
  a_b[] = value
  # Set the block, required if it is structurally zero.
  blocks(a)[] = a_b
  return a
end

function blocksparse_fill!(a::AbstractArray, value)
  for b in BlockRange(a)
    # We can't use:
    # ```julia
    # a[b] .= value
    # ```
    # since that would lead to a stack overflow,
    # because broadcasting calls `fill!`.

    # TODO: Ideally we would use:
    # ```julia
    # @view!(a[b]) .= value
    # ```
    # but that doesn't work on `SubArray` right now.

    # This line is needed to instantiate blocks
    # that aren't instantiated yet. Maybe
    # we can make this work without this line?
    blocks(a)[Int.(Tuple(b))...] = blocks(a)[Int.(Tuple(b))...]
    blocks(a)[Int.(Tuple(b))...] .= value
  end
  return a
end

function block_stored_length(a::AbstractArray)
  return stored_length(blocks(a))
end

# BlockArrays

using ..SparseArraysBase: SparseArraysBase, AbstractSparseArray, AbstractSparseMatrix

_perm(::PermutedDimsArray{<:Any,<:Any,perm}) where {perm} = perm
_invperm(::PermutedDimsArray{<:Any,<:Any,<:Any,invperm}) where {invperm} = invperm
_getindices(t::Tuple, indices) = map(i -> t[i], indices)
_getindices(i::CartesianIndex, indices) = CartesianIndex(_getindices(Tuple(i), indices))

# Represents the array of arrays of a `PermutedDimsArray`
# wrapping a block spare array, i.e. `blocks(array)` where `a` is a `PermutedDimsArray`.
struct SparsePermutedDimsArrayBlocks{
  T,N,BlockType<:AbstractArray{T,N},Array<:PermutedDimsArray{T,N}
} <: AbstractSparseArray{BlockType,N}
  array::Array
end
function blocksparse_blocks(a::PermutedDimsArray)
  return SparsePermutedDimsArrayBlocks{eltype(a),ndims(a),blocktype(parent(a)),typeof(a)}(a)
end
function Base.size(a::SparsePermutedDimsArrayBlocks)
  return _getindices(size(blocks(parent(a.array))), _perm(a.array))
end
function Base.getindex(
  a::SparsePermutedDimsArrayBlocks{<:Any,N}, index::Vararg{Int,N}
) where {N}
  return PermutedDimsArray(
    blocks(parent(a.array))[_getindices(index, _invperm(a.array))...], _perm(a.array)
  )
end
function SparseArraysBase.stored_indices(a::SparsePermutedDimsArrayBlocks)
  return map(I -> _getindices(I, _perm(a.array)), stored_indices(blocks(parent(a.array))))
end
# TODO: Either make this the generic interface or define
# `SparseArraysBase.sparse_storage`, which is used
# to defined this.
function SparseArraysBase.stored_length(a::SparsePermutedDimsArrayBlocks)
  return length(stored_indices(a))
end
function SparseArraysBase.sparse_storage(a::SparsePermutedDimsArrayBlocks)
  return error("Not implemented")
end

reverse_index(index) = reverse(index)
reverse_index(index::CartesianIndex) = CartesianIndex(reverse(Tuple(index)))

blocksparse_blocks(a::Transpose) = transpose(blocks(parent(a)))
blocksparse_blocks(a::Adjoint) = adjoint(blocks(parent(a)))

# Represents the array of arrays of a `SubArray`
# wrapping a block spare array, i.e. `blocks(array)` where `a` is a `SubArray`.
struct SparseSubArrayBlocks{T,N,BlockType<:AbstractArray{T,N},Array<:SubArray{T,N}} <:
       AbstractSparseArray{BlockType,N}
  array::Array
end
function blocksparse_blocks(a::SubArray)
  return SparseSubArrayBlocks{eltype(a),ndims(a),blocktype(parent(a)),typeof(a)}(a)
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
function Base.getindex(a::SparseSubArrayBlocks{<:Any,N}, I::Vararg{Int,N}) where {N}
  # TODO: Should this be defined as `@view a.array[Block(I)]` instead?
  return @view a.array[Block(I)]

  ## parent_blocks = @view blocks(parent(a.array))[blockrange(a)...]
  ## parent_block = parent_blocks[I...]
  ## # TODO: Define this using `blockrange(a::AbstractArray, indices::Tuple{Vararg{AbstractUnitRange}})`.
  ## block = Block(ntuple(i -> blockrange(a)[i][I[i]], ndims(a)))
  ## return @view parent_block[blockindices(parent(a.array), block, a.array.indices)...]
end
# TODO: This should be handled by generic `AbstractSparseArray` code.
function Base.getindex(a::SparseSubArrayBlocks{<:Any,N}, I::CartesianIndex{N}) where {N}
  return a[Tuple(I)...]
end
function Base.setindex!(a::SparseSubArrayBlocks{<:Any,N}, value, I::Vararg{Int,N}) where {N}
  parent_blocks = @view blocks(parent(a.array))[blockrange(a)...]
  # TODO: The following line is required to instantiate
  # uninstantiated blocks, maybe use `@view!` instead,
  # or some other code pattern.
  parent_blocks[I...] = parent_blocks[I...]
  # TODO: Define this using `blockrange(a::AbstractArray, indices::Tuple{Vararg{AbstractUnitRange}})`.
  block = Block(ntuple(i -> blockrange(a)[i][I[i]], ndims(a)))
  return parent_blocks[I...][blockindices(parent(a.array), block, a.array.indices)...] =
    value
end
function Base.isassigned(a::SparseSubArrayBlocks{<:Any,N}, I::Vararg{Int,N}) where {N}
  if CartesianIndex(I) ∉ CartesianIndices(a)
    return false
  end
  # TODO: Implement this properly.
  return true
end
function SparseArraysBase.stored_indices(a::SparseSubArrayBlocks)
  return stored_indices(view(blocks(parent(a.array)), blockrange(a)...))
end
# TODO: Either make this the generic interface or define
# `SparseArraysBase.sparse_storage`, which is used
# to defined this.
SparseArraysBase.stored_length(a::SparseSubArrayBlocks) = length(stored_indices(a))

## struct SparseSubArrayBlocksStorage{Array<:SparseSubArrayBlocks}
##   array::Array
## end
function SparseArraysBase.sparse_storage(a::SparseSubArrayBlocks)
  return map(I -> a[I], stored_indices(a))
end

function SparseArraysBase.getindex_zero_function(a::SparseSubArrayBlocks)
  # TODO: Base it off of `getindex_zero_function(blocks(parent(a.array))`, but replace the
  # axes with `axes(a.array)`.
  return BlockZero(axes(a.array))
end

to_blocks_indices(I::BlockSlice{<:BlockRange{1}}) = Int.(I.block)
to_blocks_indices(I::BlockIndices{<:Vector{<:Block{1}}}) = Int.(I.blocks)

function blocksparse_blocks(
  a::SubArray{<:Any,<:Any,<:Any,<:Tuple{Vararg{BlockSliceCollection}}}
)
  return @view blocks(parent(a))[map(to_blocks_indices, parentindices(a))...]
end

using BlockArrays: BlocksView
SparseArraysBase.stored_length(a::BlocksView) = length(a)
