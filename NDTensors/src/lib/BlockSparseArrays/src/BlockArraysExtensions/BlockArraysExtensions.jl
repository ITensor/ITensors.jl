using ArrayLayouts: ArrayLayouts, MemoryLayout, sub_materialize
using BlockArrays:
  BlockArrays,
  AbstractBlockArray,
  AbstractBlockVector,
  Block,
  BlockIndex,
  BlockIndexRange,
  BlockRange,
  BlockSlice,
  BlockVector,
  BlockedOneTo,
  BlockedUnitRange,
  BlockedVector,
  block,
  blockaxes,
  blockedrange,
  blockindex,
  blocks,
  findblock,
  findblockindex
using Compat: allequal
using Dictionaries: Dictionary, Indices
using ..GradedAxes: blockedunitrange_getindices, to_blockindices
using ..SparseArraysBase: SparseArraysBase, stored_length, stored_indices

# A return type for `blocks(array)` when `array` isn't blocked.
# Represents a vector with just that single block.
struct SingleBlockView{T,N,Array<:AbstractArray{T,N}} <: AbstractArray{T,N}
  array::Array
end
blocks_maybe_single(a) = blocks(a)
blocks_maybe_single(a::Array) = SingleBlockView(a)
function Base.getindex(a::SingleBlockView{<:Any,N}, index::Vararg{Int,N}) where {N}
  @assert all(isone, index)
  return a.array
end

# A wrapper around a potentially blocked array that is not blocked.
struct NonBlockedArray{T,N,Array<:AbstractArray{T,N}} <: AbstractArray{T,N}
  array::Array
end
Base.size(a::NonBlockedArray) = size(a.array)
Base.getindex(a::NonBlockedArray{<:Any,N}, I::Vararg{Integer,N}) where {N} = a.array[I...]
# Views of `NonBlockedArray`/`NonBlockedVector` are eager.
# This fixes an issue in Julia 1.11 where reindexing defaults to using views.
# TODO: Maybe reconsider this design, and allows views to work in slicing.
Base.view(a::NonBlockedArray, I...) = a[I...]
BlockArrays.blocks(a::NonBlockedArray) = SingleBlockView(a.array)
const NonBlockedVector{T,Array} = NonBlockedArray{T,1,Array}
NonBlockedVector(array::AbstractVector) = NonBlockedArray(array)

# BlockIndices works around an issue that the indices of BlockSlice
# are restricted to AbstractUnitRange{Int}.
struct BlockIndices{B,T<:Integer,I<:AbstractVector{T}} <: AbstractVector{T}
  blocks::B
  indices::I
end
for f in (:axes, :unsafe_indices, :axes1, :first, :last, :size, :length, :unsafe_length)
  @eval Base.$f(S::BlockIndices) = Base.$f(S.indices)
end
Base.getindex(S::BlockIndices, i::Integer) = getindex(S.indices, i)
function Base.getindex(S::BlockIndices, i::BlockSlice{<:Block{1}})
  # TODO: Check that `i.indices` is consistent with `S.indices`.
  # It seems like this isn't handling the case where `i` is a
  # subslice of a block correctly (i.e. it ignores `i.indices`).
  @assert length(S.indices[Block(i)]) == length(i.indices)
  return BlockSlice(S.blocks[Int(Block(i))], S.indices[Block(i)])
end

# This is used in slicing like:
# a = BlockSparseArray{Float64}([2, 2, 2, 2], [2, 2, 2, 2])
# I = BlockedVector([Block(4), Block(3), Block(2), Block(1)], [2, 2])
# a[I, I]
function Base.getindex(
  S::BlockIndices{<:AbstractBlockVector{<:Block{1}}}, i::BlockSlice{<:Block{1}}
)
  # TODO: Check for conistency of indices.
  # Wrapping the indices in `NonBlockedVector` reinterprets the blocked indices
  # as a single block, since the result shouldn't be blocked.
  return NonBlockedVector(BlockIndices(S.blocks[Block(i)], S.indices[Block(i)]))
end
function Base.getindex(
  S::BlockIndices{<:BlockedVector{<:Block{1},<:BlockRange{1}}}, i::BlockSlice{<:Block{1}}
)
  return i
end
# Views of `BlockIndices` are eager.
# This fixes an issue in Julia 1.11 where reindexing defaults to using views.
Base.view(S::BlockIndices, i) = S[i]

# Used in indexing such as:
# ```julia
# a = BlockSparseArray{Float64}([2, 2, 2, 2], [2, 2, 2, 2])
# I = BlockedVector([Block(4), Block(3), Block(2), Block(1)], [2, 2])
# b = @view a[I, I]
# @view b[Block(1, 1)[1:2, 2:2]]
# ```
# This is similar to the definition:
# blocksparse_to_indices(a, inds, I::Tuple{UnitRange{<:Integer},Vararg{Any}})
function Base.getindex(
  a::NonBlockedVector{<:Integer,<:BlockIndices}, I::UnitRange{<:Integer}
)
  ax = only(axes(a.array.indices))
  brs = to_blockindices(ax, I)
  inds = blockedunitrange_getindices(ax, I)
  return NonBlockedVector(a.array[BlockSlice(brs, inds)])
end

function Base.getindex(S::BlockIndices, i::BlockSlice{<:BlockRange{1}})
  # TODO: Check that `i.indices` is consistent with `S.indices`.
  # TODO: Turn this into a `blockedunitrange_getindices` definition.
  subblocks = S.blocks[Int.(i.block)]
  subindices = mortar(
    map(1:length(i.block)) do I
      r = blocks(i.indices)[I]
      return S.indices[first(r)]:S.indices[last(r)]
    end,
  )
  return BlockIndices(subblocks, subindices)
end

# Used when performing slices like:
# @views a[[Block(2), Block(1)]][2:4, 2:4]
function Base.getindex(S::BlockIndices, i::BlockSlice{<:BlockVector{<:BlockIndex{1}}})
  subblocks = mortar(
    map(blocks(i.block)) do br
      return S.blocks[Int(Block(br))][only(br.indices)]
    end,
  )
  subindices = mortar(
    map(blocks(i.block)) do br
      S.indices[br]
    end,
  )
  return BlockIndices(subblocks, subindices)
end

# Similar to the definition of `BlockArrays.BlockSlices`:
# ```julia
# const BlockSlices = Union{Base.Slice,BlockSlice{<:BlockRange{1}}}
# ```
# but includes `BlockIndices`, where the blocks aren't contiguous.
const BlockSliceCollection = Union{
  Base.Slice,BlockSlice{<:BlockRange{1}},BlockIndices{<:Vector{<:Block{1}}}
}
const SubBlockSliceCollection = BlockIndices{
  <:BlockVector{<:BlockIndex{1},<:Vector{<:BlockIndexRange{1}}}
}

# TODO: This is type piracy. This is used in `reindex` when making
# views of blocks of sliced block arrays, for example:
# ```julia
# a = BlockSparseArray{elt}(undef, ([2, 3], [2, 3]))
# b = @view a[[Block(1)[1:1], Block(2)[1:2]], [Block(1)[1:1], Block(2)[1:2]]]
# b[Block(1, 1)]
# ```
# Without this change, BlockArrays has the slicing behavior:
# ```julia
# julia> mortar([Block(1)[1:1], Block(2)[1:2]])[BlockSlice(Block(2), 2:3)]
# 2-element Vector{BlockIndex{1, Tuple{Int64}, Tuple{Int64}}}:
#  Block(2)[1]
#  Block(2)[2]
# ```
# while with this change it has the slicing behavior:
# ```julia
# julia> mortar([Block(1)[1:1], Block(2)[1:2]])[BlockSlice(Block(2), 2:3)]
# Block(2)[1:2]
# ```
# i.e. it preserves the types of the blocks better. Upstream this fix to
# BlockArrays.jl. Also consider overloading `reindex` so that it calls
# a custom `getindex` function to avoid type piracy in the meantime.
# Also fix this in BlockArrays:
# ```julia
# julia> mortar([Block(1)[1:1], Block(2)[1:2]])[Block(2)]
# 2-element Vector{BlockIndex{1, Tuple{Int64}, Tuple{Int64}}}:
#  Block(2)[1]
#  Block(2)[2]
# ```
function Base.getindex(
  a::BlockVector{<:BlockIndex{1},<:AbstractVector{<:BlockIndexRange{1}}},
  I::BlockSlice{<:Block{1}},
)
  # Check that the block slice corresponds to the correct block.
  @assert I.indices == only(axes(a))[Block(I)]
  return blocks(a)[Int(Block(I))]
end

# Outputs a `BlockUnitRange`.
function sub_axis(a::AbstractUnitRange, indices)
  return error("Not implemented")
end

# TODO: Use `GradedAxes.blockedunitrange_getindices`.
# Outputs a `BlockUnitRange`.
function sub_axis(a::AbstractUnitRange, indices::AbstractUnitRange)
  return only(axes(blockedunitrange_getindices(a, indices)))
end

# TODO: Use `GradedAxes.blockedunitrange_getindices`.
# Outputs a `BlockUnitRange`.
function sub_axis(a::AbstractUnitRange, indices::BlockSlice{<:BlockRange{1}})
  return sub_axis(a, indices.block)
end

# TODO: Use `GradedAxes.blockedunitrange_getindices`.
# Outputs a `BlockUnitRange`.
function sub_axis(a::AbstractUnitRange, indices::BlockSlice{<:Block{1}})
  return sub_axis(a, Block(indices))
end

# TODO: Use `GradedAxes.blockedunitrange_getindices`.
# Outputs a `BlockUnitRange`.
function sub_axis(a::AbstractUnitRange, indices::BlockSlice{<:BlockIndexRange{1}})
  return sub_axis(a, indices.block)
end

function sub_axis(a::AbstractUnitRange, indices::BlockIndices)
  return sub_axis(a, indices.blocks)
end

# TODO: Use `GradedAxes.blockedunitrange_getindices`.
# Outputs a `BlockUnitRange`.
function sub_axis(a::AbstractUnitRange, indices::Block)
  return only(axes(blockedunitrange_getindices(a, indices)))
end

# TODO: Use `GradedAxes.blockedunitrange_getindices`.
# Outputs a `BlockUnitRange`.
function sub_axis(a::AbstractUnitRange, indices::BlockIndexRange)
  return only(axes(blockedunitrange_getindices(a, indices)))
end

# TODO: Use `GradedAxes.blockedunitrange_getindices`.
# Outputs a `BlockUnitRange`.
function sub_axis(a::AbstractUnitRange, indices::AbstractVector{<:Block})
  return blockedrange([length(a[index]) for index in indices])
end

# TODO: Use `GradedAxes.blockedunitrange_getindices`.
# TODO: Merge blocks.
function sub_axis(a::AbstractUnitRange, indices::BlockVector{<:Block})
  # `collect` is needed here, otherwise a `PseudoBlockVector` is
  # constructed.
  return blockedrange([length(a[index]) for index in collect(indices)])
end

# TODO: Use `Tuple` conversion once
# BlockArrays.jl PR is merged.
block_to_cartesianindex(b::Block) = CartesianIndex(b.n)

function blocks_to_cartesianindices(i::Indices{<:Block})
  return block_to_cartesianindex.(i)
end

function blocks_to_cartesianindices(d::Dictionary{<:Block})
  return Dictionary(blocks_to_cartesianindices(eachindex(d)), d)
end

function block_reshape(a::AbstractArray, dims::Tuple{Vararg{Vector{Int}}})
  return block_reshape(a, blockedrange.(dims))
end

function block_reshape(a::AbstractArray, dims::Vararg{Vector{Int}})
  return block_reshape(a, dims)
end

tuple_oneto(n) = ntuple(identity, n)

function block_reshape(a::AbstractArray, axes::Tuple{Vararg{AbstractUnitRange}})
  reshaped_blocks_a = reshape(blocks(a), blocklength.(axes))
  reshaped_a = similar(a, axes)
  for I in stored_indices(reshaped_blocks_a)
    block_size_I = map(i -> length(axes[i][Block(I[i])]), tuple_oneto(length(axes)))
    # TODO: Better converter here.
    reshaped_a[Block(Tuple(I))] = reshape(reshaped_blocks_a[I], block_size_I)
  end
  return reshaped_a
end

function block_reshape(a::AbstractArray, axes::Vararg{AbstractUnitRange})
  return block_reshape(a, axes)
end

function cartesianindices(axes::Tuple, b::Block)
  return CartesianIndices(ntuple(dim -> axes[dim][Tuple(b)[dim]], length(axes)))
end

# Get the range within a block.
function blockindexrange(axis::AbstractUnitRange, r::AbstractUnitRange)
  bi1 = findblockindex(axis, first(r))
  bi2 = findblockindex(axis, last(r))
  b = block(bi1)
  # Range must fall within a single block.
  @assert b == block(bi2)
  i1 = blockindex(bi1)
  i2 = blockindex(bi2)
  return b[i1:i2]
end

function blockindexrange(
  axes::Tuple{Vararg{AbstractUnitRange,N}}, I::CartesianIndices{N}
) where {N}
  brs = blockindexrange.(axes, I.indices)
  b = Block(block.(brs))
  rs = map(br -> only(br.indices), brs)
  return b[rs...]
end

function blockindexrange(a::AbstractArray, I::CartesianIndices)
  return blockindexrange(axes(a), I)
end

# Get the blocks the range spans across.
function blockrange(axis::AbstractUnitRange, r::UnitRange)
  return findblock(axis, first(r)):findblock(axis, last(r))
end

# Occurs when slicing with `a[2:4, 2:4]`.
function blockrange(axis::BlockedOneTo{<:Integer}, r::BlockedUnitRange{<:Integer})
  # TODO: Check the blocks are commensurate.
  return findblock(axis, first(r)):findblock(axis, last(r))
end

function blockrange(axis::AbstractUnitRange, r::Int)
  ## return findblock(axis, r)
  return error("Slicing with integer values isn't supported.")
end

function blockrange(axis::AbstractUnitRange, r::AbstractVector{<:Block{1}})
  for b in r
    @assert b ∈ blockaxes(axis, 1)
  end
  return r
end

# This handles changing the blocking, for example:
# a = BlockSparseArray{Float64}([2, 2, 2, 2], [2, 2, 2, 2])
# I = blockedrange([4, 4])
# a[I, I]
# TODO: Generalize to `AbstractBlockedUnitRange`.
function blockrange(axis::BlockedOneTo{<:Integer}, r::BlockedOneTo{<:Integer})
  # TODO: Probably this is incorrect and should be something like:
  # return findblock(axis, first(r)):findblock(axis, last(r))
  return only(blockaxes(r))
end

# This handles block merging:
# a = BlockSparseArray{Float64}([2, 2, 2, 2], [2, 2, 2, 2])
# I = BlockedVector(Block.(1:4), [2, 2])
# I = BlockVector(Block.(1:4), [2, 2])
# I = BlockedVector([Block(4), Block(3), Block(2), Block(1)], [2, 2])
# I = BlockVector([Block(4), Block(3), Block(2), Block(1)], [2, 2])
# a[I, I]
function blockrange(axis::BlockedOneTo{<:Integer}, r::AbstractBlockVector{<:Block{1}})
  for b in r
    @assert b ∈ blockaxes(axis, 1)
  end
  return only(blockaxes(r))
end

using BlockArrays: BlockSlice
function blockrange(axis::AbstractUnitRange, r::BlockSlice)
  return blockrange(axis, r.block)
end

function blockrange(a::AbstractUnitRange, r::BlockIndices)
  return blockrange(a, r.blocks)
end

function blockrange(axis::AbstractUnitRange, r::Block{1})
  return r:r
end

function blockrange(axis::AbstractUnitRange, r::BlockIndexRange)
  return Block(r):Block(r)
end

function blockrange(axis::AbstractUnitRange, r::AbstractVector{<:BlockIndexRange{1}})
  return error("Slicing not implemented for range of type `$(typeof(r))`.")
end

function blockrange(
  axis::AbstractUnitRange,
  r::BlockVector{<:BlockIndex{1},<:AbstractVector{<:BlockIndexRange{1}}},
)
  return map(b -> Block(b), blocks(r))
end

# This handles slicing with `:`/`Colon()`.
function blockrange(axis::AbstractUnitRange, r::Base.Slice)
  # TODO: Maybe use `BlockRange`, but that doesn't output
  # the same thing.
  return only(blockaxes(axis))
end

function blockrange(axis::AbstractUnitRange, r::NonBlockedVector)
  return Block(1):Block(1)
end

function blockrange(axis::AbstractUnitRange, r)
  return error("Slicing not implemented for range of type `$(typeof(r))`.")
end

# This takes a range of indices `indices` of array `a`
# and maps it to the range of indices within block `block`.
function blockindices(a::AbstractArray, block::Block, indices::Tuple)
  return blockindices(axes(a), block, indices)
end

function blockindices(axes::Tuple, block::Block, indices::Tuple)
  return blockindices.(axes, Tuple(block), indices)
end

function blockindices(axis::AbstractUnitRange, block::Block, indices::AbstractUnitRange)
  indices_within_block = intersect(indices, axis[block])
  if iszero(length(indices_within_block))
    # Falls outside of block
    return 1:0
  end
  return only(blockindexrange(axis, indices_within_block).indices)
end

# This catches the case of `Vector{<:Block{1}}`.
# `BlockRange` gets wrapped in a `BlockSlice`, which is handled properly
#  by the version with `indices::AbstractUnitRange`.
#  TODO: This should get fixed in a better way inside of `BlockArrays`.
function blockindices(
  axis::AbstractUnitRange, block::Block, indices::AbstractVector{<:Block{1}}
)
  if block ∉ indices
    # Falls outside of block
    return 1:0
  end
  return Base.OneTo(length(axis[block]))
end

function blockindices(a::AbstractUnitRange, b::Block, r::BlockIndices)
  return blockindices(a, b, r.blocks)
end

function blockindices(
  a::AbstractUnitRange,
  b::Block,
  r::BlockVector{<:BlockIndex{1},<:AbstractVector{<:BlockIndexRange{1}}},
)
  # TODO: Change to iterate over `BlockRange(r)`
  # once https://github.com/JuliaArrays/BlockArrays.jl/issues/404
  # is fixed.
  for bl in blocks(r)
    if b == Block(bl)
      return only(bl.indices)
    end
  end
  return error("Block not found.")
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

_block(indices) = block(indices)
_block(indices::CartesianIndices) = Block(ntuple(Returns(1), ndims(indices)))

function combine_axes(as::Vararg{Tuple})
  @assert allequal(length.(as))
  ndims = length(first(as))
  return ntuple(ndims) do dim
    dim_axes = map(a -> a[dim], as)
    return reduce(BlockArrays.combine_blockaxes, dim_axes)
  end
end

# Returns `BlockRange`
# Convert the block of the axes to blocks of the subaxes.
function subblocks(axes::Tuple, subaxes::Tuple, block::Block)
  @assert length(axes) == length(subaxes)
  return BlockRange(
    ntuple(length(axes)) do dim
      findblocks(subaxes[dim], axes[dim][Tuple(block)[dim]])
    end,
  )
end

# Returns `Vector{<:Block}`
function subblocks(axes::Tuple, subaxes::Tuple, blocks)
  return mapreduce(vcat, blocks; init=eltype(blocks)[]) do block
    return vec(subblocks(axes, subaxes, block))
  end
end

# Returns `Vector{<:CartesianIndices}`
function blocked_cartesianindices(axes::Tuple, subaxes::Tuple, blocks)
  return map(subblocks(axes, subaxes, blocks)) do block
    return cartesianindices(subaxes, block)
  end
end

# Represents a view of a block of a blocked array.
struct BlockView{T,N,Array<:AbstractArray{T,N}} <: AbstractArray{T,N}
  array::Array
  block::Tuple{Vararg{Block{1,Int},N}}
end
function Base.axes(a::BlockView)
  # TODO: Try to avoid conversion to `Base.OneTo{Int}`, or just convert
  # the element type to `Int` with `Int.(...)`.
  # When the axes of `a.array` are `GradedOneTo`, the block is `LabelledUnitRange`,
  # which has element type `LabelledInteger`. That causes conversion problems
  # in some generic Base Julia code, for example when printing `BlockView`.
  return ntuple(ndims(a)) do dim
    return Base.OneTo{Int}(only(axes(axes(a.array, dim)[a.block[dim]])))
  end
end
function Base.size(a::BlockView)
  return length.(axes(a))
end
function Base.getindex(a::BlockView{<:Any,N}, index::Vararg{Int,N}) where {N}
  return blocks(a.array)[Int.(a.block)...][index...]
end
function Base.setindex!(a::BlockView{<:Any,N}, value, index::Vararg{Int,N}) where {N}
  blocks(a.array)[Int.(a.block)...] = blocks(a.array)[Int.(a.block)...]
  blocks(a.array)[Int.(a.block)...][index...] = value
  return a
end

function SparseArraysBase.stored_length(a::BlockView)
  # TODO: Store whether or not the block is stored already as
  # a Bool in `BlockView`.
  I = CartesianIndex(Int.(a.block))
  # TODO: Use `block_stored_indices`.
  if I ∈ stored_indices(blocks(a.array))
    return stored_length(blocks(a.array)[I])
  end
  return 0
end

## # Allow more fine-grained control:
## function ArrayLayouts.sub_materialize(layout, a::BlockView, ax)
##   return blocks(a.array)[Int.(a.block)...]
## end
## function ArrayLayouts.sub_materialize(layout, a::BlockView)
##   return sub_materialize(layout, a, axes(a))
## end
## function ArrayLayouts.sub_materialize(a::BlockView)
##   return sub_materialize(MemoryLayout(a), a)
## end
function ArrayLayouts.sub_materialize(a::BlockView)
  return blocks(a.array)[Int.(a.block)...]
end

function view!(a::AbstractArray{<:Any,N}, index::Block{N}) where {N}
  return view!(a, Tuple(index)...)
end
function view!(a::AbstractArray{<:Any,N}, index::Vararg{Block{1},N}) where {N}
  blocks(a)[Int.(index)...] = blocks(a)[Int.(index)...]
  return blocks(a)[Int.(index)...]
end

function view!(a::AbstractArray{<:Any,N}, index::BlockIndexRange{N}) where {N}
  # TODO: Is there a better code pattern for this?
  indices = ntuple(N) do dim
    return Tuple(Block(index))[dim][index.indices[dim]]
  end
  return view!(a, indices...)
end
function view!(a::AbstractArray{<:Any,N}, index::Vararg{BlockIndexRange{1},N}) where {N}
  b = view!(a, Block.(index)...)
  r = map(index -> only(index.indices), index)
  return @view b[r...]
end

using MacroTools: @capture
using NDTensors.SparseArraysBase: is_getindex_expr
macro view!(expr)
  if !is_getindex_expr(expr)
    error("@view must be used with getindex syntax (as `@view! a[i,j,...]`)")
  end
  @capture(expr, array_[indices__])
  return :(view!($(esc(array)), $(esc.(indices)...)))
end
