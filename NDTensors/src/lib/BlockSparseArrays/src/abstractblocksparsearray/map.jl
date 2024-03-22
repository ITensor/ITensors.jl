using ArrayLayouts: LayoutArray
using BlockArrays: blockisequal
using ..SparseArrayInterface:
  SparseArrayInterface,
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
  stored_blocked_cartesianindices_as = map(as) do a
    return blocked_cartesianindices(
      axes(a), combine_axes(axes.(as)...), block_stored_indices(a)
    )
  end
  return ∪(stored_blocked_cartesianindices_as...)
end

# This is used by `map` to get the output axes.
# This is type piracy, try to avoid this, maybe requires defining `map`.
## Base.promote_shape(a1::Tuple{Vararg{BlockedUnitRange}}, a2::Tuple{Vararg{BlockedUnitRange}}) = combine_axes(a1, a2)

function SparseArrayInterface.sparse_map!(
  ::BlockSparseArrayStyle, f, a_dest::AbstractArray, a_srcs::Vararg{AbstractArray}
)
  for I in union_stored_blocked_cartesianindices(a_dest, a_srcs...)
    BI_dest = blockindexrange(a_dest, I)
    BI_srcs = map(a_src -> blockindexrange(a_src, I), a_srcs)
    block_dest = @view a_dest[_block(BI_dest)]
    block_srcs = ntuple(i -> @view(a_srcs[i][_block(BI_srcs[i])]), length(a_srcs))
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
# function SparseArrayInterface.sparse_mapreduce(::BlockSparseArrayStyle, f, a_dest::AbstractArray, a_srcs::Vararg{AbstractArray})
# end

function Base.map!(f, a_dest::AbstractArray, a_srcs::Vararg{BlockSparseArrayLike})
  sparse_map!(f, a_dest, a_srcs...)
  return a_dest
end

function Base.map(f, as::Vararg{BlockSparseArrayLike})
  return f.(as...)
end

function Base.copy!(a_dest::AbstractArray, a_src::BlockSparseArrayLike)
  sparse_copy!(a_dest, a_src)
  return a_dest
end

function Base.copyto!(a_dest::AbstractArray, a_src::BlockSparseArrayLike)
  sparse_copyto!(a_dest, a_src)
  return a_dest
end

# Fix ambiguity error
function Base.copyto!(a_dest::LayoutArray, a_src::BlockSparseArrayLike)
  sparse_copyto!(a_dest, a_src)
  return a_dest
end

function Base.permutedims!(a_dest, a_src::BlockSparseArrayLike, perm)
  sparse_permutedims!(a_dest, a_src, perm)
  return a_dest
end

function Base.mapreduce(f, op, as::Vararg{BlockSparseArrayLike}; kwargs...)
  return sparse_mapreduce(f, op, as...; kwargs...)
end

# TODO: Why isn't this calling `mapreduce` already?
function Base.iszero(a::BlockSparseArrayLike)
  return sparse_iszero(a)
end

# TODO: Why isn't this calling `mapreduce` already?
function Base.isreal(a::BlockSparseArrayLike)
  return sparse_isreal(a)
end
