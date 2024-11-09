module BlockSparseArraysGradedAxesExt
using BlockArrays:
  AbstractBlockVector,
  AbstractBlockedUnitRange,
  Block,
  BlockIndexRange,
  blockedrange,
  blocks
using ..BlockSparseArrays:
  BlockSparseArrays,
  AbstractBlockSparseArray,
  AbstractBlockSparseMatrix,
  BlockSparseArray,
  BlockSparseMatrix,
  BlockSparseVector,
  block_merge
using ...GradedAxes:
  GradedAxes,
  AbstractGradedUnitRange,
  OneToOne,
  blockmergesortperm,
  blocksortperm,
  dual,
  invblockperm,
  nondual,
  tensor_product
using LinearAlgebra: Adjoint, Transpose
using ...TensorAlgebra:
  TensorAlgebra, FusionStyle, BlockReshapeFusion, SectorFusion, fusedims, splitdims

# TODO: Make a `ReduceWhile` library.
include("reducewhile.jl")

TensorAlgebra.FusionStyle(::AbstractGradedUnitRange) = SectorFusion()

# TODO: Need to implement this! Will require implementing
# `block_merge(a::AbstractUnitRange, blockmerger::BlockedUnitRange)`.
function BlockSparseArrays.block_merge(
  a::AbstractGradedUnitRange, blockmerger::AbstractBlockedUnitRange
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
  # TODO: This is doing extra copies of the blocks,
  # use `@view a[axes_prod...]` instead.
  # That will require implementing some reindexing logic
  # for this combination of slicing.
  a_unblocked = a[axes_prod...]
  a_blockpermed = a_unblocked[blockperms...]
  return splitdims(BlockReshapeFusion(), a_blockpermed, split_axes...)
end

# This is a temporary fix for `eachindex` being broken for BlockSparseArrays
# with mixed dual and non-dual axes. This shouldn't be needed once
# GradedAxes is rewritten using BlockArrays v1.
# TODO: Delete this once GradedAxes is rewritten.
function Base.eachindex(a::AbstractBlockSparseArray)
  return CartesianIndices(nondual.(axes(a)))
end

# TODO: Handle this through some kind of trait dispatch, maybe
# a `SymmetryStyle`-like trait to check if the block sparse
# matrix has graded axes.
function Base.axes(a::Adjoint{<:Any,<:AbstractBlockSparseMatrix})
  return dual.(reverse(axes(a')))
end

# This definition is only needed since calls like
# `a[[Block(1), Block(2)]]` where `a isa AbstractGradedUnitRange`
# returns a `BlockSparseVector` instead of a `BlockVector`
# due to limitations in the `BlockArray` type not allowing
# axes with non-Int element types.
# TODO: Remove this once that issue is fixed,
# see https://github.com/JuliaArrays/BlockArrays.jl/pull/405.
using BlockArrays: BlockRange
using NDTensors.LabelledNumbers: label
function GradedAxes.blocklabels(a::BlockSparseVector)
  return map(BlockRange(a)) do block
    return label(blocks(a)[Int(block)])
  end
end

# This is a temporary fix for `show` being broken for BlockSparseArrays
# with mixed dual and non-dual axes. This shouldn't be needed once
# GradedAxes is rewritten using BlockArrays v1.
# TODO: Delete this once GradedAxes is rewritten.
function blocksparse_show(
  io::IO, mime::MIME"text/plain", a::AbstractArray, axes_a::Tuple; kwargs...
)
  println(io, "typeof(axes) = ", typeof(axes_a), "\n")
  println(
    io,
    "Warning: To temporarily circumvent a bug in printing BlockSparseArrays with mixtures of dual and non-dual axes, the types of the dual axes printed below might not be accurate. The types printed above this message are the correct ones.\n",
  )
  return invoke(show, Tuple{IO,MIME"text/plain",AbstractArray}, io, mime, a; kwargs...)
end

# This is a temporary fix for `show` being broken for BlockSparseArrays
# with mixed dual and non-dual axes. This shouldn't be needed once
# GradedAxes is rewritten using BlockArrays v1.
# TODO: Delete this once GradedAxes is rewritten.
function Base.show(io::IO, mime::MIME"text/plain", a::BlockSparseArray; kwargs...)
  axes_a = axes(a)
  a_nondual = BlockSparseArray(blocks(a), nondual.(axes(a)))
  return blocksparse_show(io, mime, a_nondual, axes_a; kwargs...)
end

# This is a temporary fix for `show` being broken for BlockSparseArrays
# with mixed dual and non-dual axes. This shouldn't be needed once
# GradedAxes is rewritten using BlockArrays v1.
# TODO: Delete this once GradedAxes is rewritten.
function Base.show(
  io::IO, mime::MIME"text/plain", a::Adjoint{<:Any,<:BlockSparseMatrix}; kwargs...
)
  axes_a = axes(a)
  a_nondual = BlockSparseArray(blocks(a'), dual.(nondual.(axes(a'))))'
  return blocksparse_show(io, mime, a_nondual, axes_a; kwargs...)
end

# This is a temporary fix for `show` being broken for BlockSparseArrays
# with mixed dual and non-dual axes. This shouldn't be needed once
# GradedAxes is rewritten using BlockArrays v1.
# TODO: Delete this once GradedAxes is rewritten.
function Base.show(
  io::IO, mime::MIME"text/plain", a::Transpose{<:Any,<:BlockSparseMatrix}; kwargs...
)
  axes_a = axes(a)
  a_nondual = tranpose(BlockSparseArray(transpose(blocks(a)), nondual.(axes(a))))
  return blocksparse_show(io, mime, a_nondual, axes_a; kwargs...)
end
end
