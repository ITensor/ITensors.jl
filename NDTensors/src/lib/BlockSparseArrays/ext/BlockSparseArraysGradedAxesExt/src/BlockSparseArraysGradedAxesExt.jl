module BlockSparseArraysGradedAxesExt
using BlockArrays: AbstractBlockVector, Block, BlockedUnitRange
using ..BlockSparseArrays: BlockSparseArrays, block_merge
using ...GradedAxes:
  AbstractGradedUnitRange,
  OneToOne,
  blockmergesortperm,
  blocksortperm,
  invblockperm,
  tensor_product
using ...TensorAlgebra:
  TensorAlgebra, FusionStyle, BlockReshapeFusion, SectorFusion, fusedims, splitdims

# TODO: Make a `ReduceWhile` library.
include("reducewhile.jl")

TensorAlgebra.FusionStyle(::AbstractGradedUnitRange) = SectorFusion()

# TODO: Need to implement this! Will require implementing
# `block_merge(a::AbstractUnitRange, blockmerger::BlockedUnitRange)`.
function BlockSparseArrays.block_merge(
  a::AbstractGradedUnitRange, blockmerger::BlockedUnitRange
)
  return a
end

# Sort the blocks by sector and then merge the common sectors.
function block_mergesort(a::AbstractArray)
  I = blockmergesortperm.(axes(a))
  return a[I...]
end

function TensorAlgebra.fusedims(
  ::SectorFusion, a::AbstractArray, axes::AbstractUnitRange...
)
  # First perform a fusion using a block reshape.
  a_reshaped = fusedims(BlockReshapeFusion(), a, axes...)
  # Sort the blocks by sector and merge the equivalent sectors.
  return block_mergesort(a_reshaped)
end

function TensorAlgebra.splitdims(
  ::SectorFusion, a::AbstractArray, split_axes::AbstractUnitRange...
)
  # First, fuse axes to get `blockmergesortperm`.
  # Then unpermute the blocks.
  axes_prod =
    groupreducewhile(tensor_product, split_axes, ndims(a); init=OneToOne()) do i, axis
      return length(axis) ≤ length(axes(a, i))
    end
  blockperms = invblockperm.(blocksortperm.(axes_prod))
  a_blockpermed = a[blockperms...]
  return splitdims(BlockReshapeFusion(), a_blockpermed, split_axes...)
end
end
