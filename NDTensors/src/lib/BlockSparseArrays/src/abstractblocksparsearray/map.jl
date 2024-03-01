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

## using BlockArrays: BlockArrays, BlockRange, block

## _block(indices) = block(indices)
## _block(indices::CartesianIndices) = Block(ntuple(Returns(1), ndims(indices)))
##
## function combine_axes(as::Vararg{Tuple})
##   @assert allequal(length.(as))
##   ndims = length(first(as))
##   return ntuple(ndims) do dim
##     dim_axes = map(a -> a[dim], as)
##     return reduce(BlockArrays.combine_blockaxes, dim_axes)
##   end
## end
##
## # Returns `BlockRange`
## # Convert the block of the axes to blocks of the subaxes.
## function subblocks(axes::Tuple, subaxes::Tuple, block::Block)
##   @assert length(axes) == length(subaxes)
##   return BlockRange(
##     ntuple(length(axes)) do dim
##       findblocks(subaxes[dim], axes[dim][Tuple(block)[dim]])
##     end,
##   )
## end
##
## # Returns `Vector{<:Block}`
## function subblocks(axes::Tuple, subaxes::Tuple, blocks)
##   return mapreduce(vcat, blocks; init=eltype(blocks)[]) do block
##     return vec(subblocks(axes, subaxes, block))
##   end
## end
##
## # Returns `Vector{<:CartesianIndices}`
## function stored_blocked_cartesianindices(a::AbstractArray, subaxes::Tuple)
##   return map(subblocks(axes(a), subaxes, block_stored_indices(a))) do block
##     return cartesianindices(subaxes, block)
##   end
## end

# Returns `Vector{<:CartesianIndices}`
function union_stored_blocked_cartesianindices(as::Vararg{AbstractArray})
  stored_blocked_cartesianindices_as = map(as) do a
    return blocked_cartesianindices(
      axes(a), combine_axes(axes.(as)...), block_stored_indices(a)
    )
  end
  return ∪(stored_blocked_cartesianindices_as...)
end

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

# Map
function Base.map!(f, a_dest::AbstractArray, a_srcs::Vararg{BlockSparseArrayLike})
  sparse_map!(f, a_dest, a_srcs...)
  return a_dest
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
