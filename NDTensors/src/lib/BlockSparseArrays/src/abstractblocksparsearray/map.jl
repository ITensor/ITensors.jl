using ArrayLayouts: LayoutArray

# Map
function Base.map!(f, a_dest::AbstractArray, a_srcs::Vararg{BlockSparseArrayLike})
  blocksparse_map!(f, a_dest, a_srcs...)
  return a_dest
end

function Base.copy!(a_dest::AbstractArray, a_src::BlockSparseArrayLike)
  blocksparse_copy!(a_dest, a_src)
  return a_dest
end

function Base.copyto!(a_dest::AbstractArray, a_src::BlockSparseArrayLike)
  blocksparse_copyto!(a_dest, a_src)
  return a_dest
end

# Fix ambiguity error
function Base.copyto!(a_dest::LayoutArray, a_src::BlockSparseArrayLike)
  blocksparse_copyto!(a_dest, a_src)
  return a_dest
end

function Base.permutedims!(a_dest::AbstractArray, a_src::BlockSparseArrayLike, perm)
  blocksparse_permutedims!(a_dest, a_src, perm)
  return a_dest
end

function Base.mapreduce(f, op, as::Vararg{BlockSparseArrayLike}; kwargs...)
  return blocksparse_mapreduce(f, op, as...; kwargs...)
end

# TODO: Why isn't this calling `mapreduce` already?
function Base.iszero(a::BlockSparseArrayLike)
  return blocksparse_iszero(a)
end

# TODO: Why isn't this calling `mapreduce` already?
function Base.isreal(a::BlockSparseArrayLike)
  return blocksparse_isreal(a)
end
