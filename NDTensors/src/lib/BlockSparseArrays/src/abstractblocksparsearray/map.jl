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

function SparseArrayInterface.sparse_map!(
  ::BlockSparseArrayStyle, f, a_dest::AbstractArray, a_srcs::Vararg{AbstractArray}
)
  if all(a_src -> blockisequal(axes(a_dest), axes(a_src)), a_srcs)
    # If the axes/block structure are all the same,
    # map based on the blocks.
    map!(f, blocks(a_dest), blocks.(a_srcs)...)
  else
    map_mismatched_blocking!(f, a_dest, a_srcs...)
  end
  return a_dest
end

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
