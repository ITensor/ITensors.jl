using BlockArrays: BlockArrays

# TODO: Delete this.
using BlockArrays: blocks

struct BlockSparseArray{
  T,
  N,
  A<:AbstractArray{T,N},
  Blocks<:AbstractArray{A,N},
  Axes<:NTuple{N,AbstractUnitRange{Int}},
} <: AbstractBlockSparseArray{T,N}
  blocks::Blocks
  axes::Axes
end

# Base `AbstractArray` interface
Base.axes(a::BlockSparseArray) = a.axes

# BlockArrays `AbstractBlockArray` interface
BlockArrays.blocks(::BlockSparseArray) = a.blocks

# `AbstractBlockSparseArray` interface
# TODO: Use `SetParameters`.
blocktype(::BlockSparseArray{<:Any,<:Any,A}) where {A} = A
