using ArrayLayouts: LayoutArray
using BlockArrays: blockisequal
using LinearAlgebra: Adjoint, Transpose
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
  combined_axes = combine_axes(axes.(as)...)
  stored_blocked_cartesianindices_as = map(as) do a
    return blocked_cartesianindices(axes(a), combined_axes, block_stored_indices(a))
  end
  return ∪(stored_blocked_cartesianindices_as...)
end

# This is used by `map` to get the output axes.
# This is type piracy, try to avoid this, maybe requires defining `map`.
## Base.promote_shape(a1::Tuple{Vararg{BlockedUnitRange}}, a2::Tuple{Vararg{BlockedUnitRange}}) = combine_axes(a1, a2)

struct SingleBlockView{T,N,Array<:AbstractArray{T,N}} <: AbstractArray{T,N}
  array::Array
end
_blocks(a) = blocks(a)
_blocks(a::Array) = SingleBlockView(a)
function Base.getindex(a::SingleBlockView{<:Any,N}, index::Vararg{Int,N}) where {N}
  @assert all(isone, index)
  return a.array
end

reblock(a) = a
function reblock(a::SubArray{<:Any,<:Any,<:Any,<:Tuple{Vararg{BlockSlice}}})
  return @view a.parent[map(i -> i.indices, a.indices)...]
end

function SparseArrayInterface.sparse_map!(
  ::BlockSparseArrayStyle, f, a_dest::AbstractArray, a_srcs::Vararg{AbstractArray}
)
  a_dest, a_srcs = reblock(a_dest), reblock.(a_srcs)
  for I in union_stored_blocked_cartesianindices(a_dest, a_srcs...)
    BI_dest = blockindexrange(a_dest, I)
    BI_srcs = map(a_src -> blockindexrange(a_src, I), a_srcs)
    # TODO: Investigate why this doesn't work:
    # block_dest = @view a_dest[_block(BI_dest)]
    block_dest = _blocks(a_dest)[Int.(Tuple(_block(BI_dest)))...]
    # TODO: Investigate why this doesn't work:
    # block_srcs = ntuple(i -> @view(a_srcs[i][_block(BI_srcs[i])]), length(a_srcs))
    block_srcs = ntuple(length(a_srcs)) do i
      return blocks(a_srcs[i])[Int.(Tuple(_block(BI_srcs[i])))...]
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

function Base.permutedims!(a_dest, a_src::BlockSparseArrayLike, perm)
  sparse_permutedims!(a_dest, a_src, perm)
  return a_dest
end

function Base.mapreduce(f, op, as::Vararg{BlockSparseArrayLike}; kwargs...)
  return sparse_mapreduce(f, op, as...; kwargs...)
end

# TODO: Why isn't this calling `mapreduce` already?
function Base.iszero(a::BlockSparseArrayLike)
  return sparse_iszero(blocks(a))
end

# TODO: Why isn't this calling `mapreduce` already?
function Base.isreal(a::BlockSparseArrayLike)
  return sparse_isreal(blocks(a))
end
