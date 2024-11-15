using ArrayLayouts: LayoutArray

# Map
function Base.map!(f, a_dest::AbstractArray, a_srcs::Vararg{AnyAbstractSparseArray})
  SparseArraysBase.sparse_map!(f, a_dest, a_srcs...)
  return a_dest
end

function Base.copy!(a_dest::AbstractArray, a_src::AnyAbstractSparseArray)
  SparseArraysBase.sparse_copy!(a_dest, a_src)
  return a_dest
end

function Base.copyto!(a_dest::AbstractArray, a_src::AnyAbstractSparseArray)
  SparseArraysBase.sparse_copyto!(a_dest, a_src)
  return a_dest
end

# Fix ambiguity error
function Base.copyto!(a_dest::LayoutArray, a_src::AnyAbstractSparseArray)
  SparseArraysBase.sparse_copyto!(a_dest, a_src)
  return a_dest
end

function Base.permutedims!(a_dest::AbstractArray, a_src::AnyAbstractSparseArray, perm)
  SparseArraysBase.sparse_permutedims!(a_dest, a_src, perm)
  return a_dest
end

function Base.mapreduce(f, op, as::Vararg{AnyAbstractSparseArray}; kwargs...)
  return SparseArraysBase.sparse_mapreduce(f, op, as...; kwargs...)
end

# TODO: Why isn't this calling `mapreduce` already?
function Base.iszero(a::AnyAbstractSparseArray)
  return SparseArraysBase.sparse_iszero(a)
end

# TODO: Why isn't this calling `mapreduce` already?
function Base.isreal(a::AnyAbstractSparseArray)
  return SparseArraysBase.sparse_isreal(a)
end
