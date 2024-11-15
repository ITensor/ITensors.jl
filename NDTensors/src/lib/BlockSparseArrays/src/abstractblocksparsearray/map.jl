using ArrayLayouts: LayoutArray
using BlockArrays: blockisequal
using LinearAlgebra: Adjoint, Transpose
using ..SparseArraysBase:
  SparseArraysBase,
  SparseArrayStyle,
  sparse_map!,
  sparse_copy!,
  sparse_copyto!,
  sparse_permutedims!,
  sparse_mapreduce,
  sparse_iszero,
  sparse_isreal

# Returns `Vector{<:CartesianIndices}`
function union_stored_blocked_cartesianindices(as::Vararg{AbstractArray})
  combined_axes = combine_axes(axes.(as)...)
  stored_blocked_cartesianindices_as = map(as) do a
    return blocked_cartesianindices(axes(a), combined_axes, block_stored_indices(a))
  end
  return ∪(stored_blocked_cartesianindices_as...)
end

# This is used by `map` to get the output axes.
# This is type piracy, try to avoid this, maybe requires defining `map`.
## Base.promote_shape(a1::Tuple{Vararg{BlockedUnitRange}}, a2::Tuple{Vararg{BlockedUnitRange}}) = combine_axes(a1, a2)

reblock(a) = a

# If the blocking of the slice doesn't match the blocking of the
# parent array, reblock according to the blocking of the parent array.
function reblock(
  a::SubArray{<:Any,<:Any,<:AbstractBlockSparseArray,<:Tuple{Vararg{AbstractUnitRange}}}
)
  # TODO: This relies on the behavior that slicing a block sparse
  # array with a UnitRange inherits the blocking of the underlying
  # block sparse array, we might change that default behavior
  # so this might become something like `@blocked parent(a)[...]`.
  return @view parent(a)[UnitRange{Int}.(parentindices(a))...]
end

function reblock(
  a::SubArray{<:Any,<:Any,<:AbstractBlockSparseArray,<:Tuple{Vararg{NonBlockedArray}}}
)
  return @view parent(a)[map(I -> I.array, parentindices(a))...]
end

function reblock(
  a::SubArray{
    <:Any,
    <:Any,
    <:AbstractBlockSparseArray,
    <:Tuple{Vararg{BlockIndices{<:AbstractBlockVector{<:Block{1}}}}},
  },
)
  # Remove the blocking.
  return @view parent(a)[map(I -> Vector(I.blocks), parentindices(a))...]
end

# TODO: Rewrite this so that it takes the blocking structure
# made by combining the blocking of the axes (i.e. the blocking that
# is used to determine `union_stored_blocked_cartesianindices(...)`).
# `reblock` is a partial solution to that, but a bit ad-hoc.
# TODO: Move to `blocksparsearrayinterface/map.jl`.
function SparseArraysBase.sparse_map!(
  ::BlockSparseArrayStyle, f, a_dest::AbstractArray, a_srcs::Vararg{AbstractArray}
)
  a_dest, a_srcs = reblock(a_dest), reblock.(a_srcs)
  for I in union_stored_blocked_cartesianindices(a_dest, a_srcs...)
    BI_dest = blockindexrange(a_dest, I)
    BI_srcs = map(a_src -> blockindexrange(a_src, I), a_srcs)
    # TODO: Investigate why this doesn't work:
    # block_dest = @view a_dest[_block(BI_dest)]
    block_dest = blocks_maybe_single(a_dest)[Int.(Tuple(_block(BI_dest)))...]
    # TODO: Investigate why this doesn't work:
    # block_srcs = ntuple(i -> @view(a_srcs[i][_block(BI_srcs[i])]), length(a_srcs))
    block_srcs = ntuple(length(a_srcs)) do i
      return blocks_maybe_single(a_srcs[i])[Int.(Tuple(_block(BI_srcs[i])))...]
    end
    subblock_dest = @view block_dest[BI_dest.indices...]
    subblock_srcs = ntuple(i -> @view(block_srcs[i][BI_srcs[i].indices...]), length(a_srcs))
    # TODO: Use `map!!` to handle immutable blocks.
    map!(f, subblock_dest, subblock_srcs...)
    # Replace the entire block, handles initializing new blocks
    # or if blocks are immutable.
    blocks(a_dest)[Int.(Tuple(_block(BI_dest)))...] = block_dest
  end
  return a_dest
end

# TODO: Implement this.
# function SparseArraysBase.sparse_mapreduce(::BlockSparseArrayStyle, f, a_dest::AbstractArray, a_srcs::Vararg{AbstractArray})
# end

function Base.map!(f, a_dest::AbstractArray, a_srcs::Vararg{AnyAbstractBlockSparseArray})
  sparse_map!(f, a_dest, a_srcs...)
  return a_dest
end

function Base.map(f, as::Vararg{AnyAbstractBlockSparseArray})
  return f.(as...)
end

function Base.copy!(a_dest::AbstractArray, a_src::AnyAbstractBlockSparseArray)
  sparse_copy!(a_dest, a_src)
  return a_dest
end

function Base.copyto!(a_dest::AbstractArray, a_src::AnyAbstractBlockSparseArray)
  sparse_copyto!(a_dest, a_src)
  return a_dest
end

# Fix ambiguity error
function Base.copyto!(a_dest::LayoutArray, a_src::AnyAbstractBlockSparseArray)
  sparse_copyto!(a_dest, a_src)
  return a_dest
end

function Base.copyto!(
  a_dest::AbstractMatrix, a_src::Transpose{T,<:AbstractBlockSparseMatrix{T}}
) where {T}
  sparse_copyto!(a_dest, a_src)
  return a_dest
end

function Base.copyto!(
  a_dest::AbstractMatrix, a_src::Adjoint{T,<:AbstractBlockSparseMatrix{T}}
) where {T}
  sparse_copyto!(a_dest, a_src)
  return a_dest
end

function Base.permutedims!(a_dest, a_src::AnyAbstractBlockSparseArray, perm)
  sparse_permutedims!(a_dest, a_src, perm)
  return a_dest
end

function Base.mapreduce(f, op, as::Vararg{AnyAbstractBlockSparseArray}; kwargs...)
  return sparse_mapreduce(f, op, as...; kwargs...)
end

# TODO: Why isn't this calling `mapreduce` already?
function Base.iszero(a::AnyAbstractBlockSparseArray)
  return sparse_iszero(blocks(a))
end

# TODO: Why isn't this calling `mapreduce` already?
function Base.isreal(a::AnyAbstractBlockSparseArray)
  return sparse_isreal(blocks(a))
end
