using BlockArrays:
  BlockArrays,
  AbstractBlockArray,
  AbstractBlockVector,
  Block,
  BlockRange,
  BlockedUnitRange,
  BlockVector,
  BlockSlice,
  block,
  blockaxes,
  blockedrange,
  blockindex,
  blocks,
  findblock,
  findblockindex
using Compat: allequal
using Dictionaries: Dictionary, Indices
using ..GradedAxes: blockedunitrange_getindices
using ..SparseArrayInterface: stored_indices

struct BlockIndices{B,I<:AbstractVector{Int}} <: AbstractVector{Int}
  blocks::B
  indices::I
end
for f in (:axes, :unsafe_indices, :axes1, :first, :last, :size, :length, :unsafe_length)
  @eval Base.$f(S::BlockIndices) = Base.$f(S.indices)
end
Base.getindex(S::BlockIndices, i::Integer) = getindex(S.indices, i)

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
function blockindexrange(axis::AbstractUnitRange, r::UnitRange)
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

function blockrange(axis::AbstractUnitRange, r::Int)
  error("Slicing with integer values isn't supported.")
  return findblock(axis, r)
end

function blockrange(axis::AbstractUnitRange, r::AbstractVector{<:Block{1}})
  for b in r
    @assert b ∈ blockaxes(axis, 1)
  end
  return r
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
