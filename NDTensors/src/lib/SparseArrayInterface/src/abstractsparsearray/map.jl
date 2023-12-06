using ArrayLayouts: LayoutArray

# Map
function Base.map!(f, dest::AbstractArray, src::AbstractSparseArray)
  SparseArrayInterface.sparse_map!(f, dest, src)
  return dest
end

function Base.copy!(dest::AbstractArray, src::AbstractSparseArray)
  SparseArrayInterface.sparse_copy!(dest, src)
  return dest
end

function Base.copyto!(dest::AbstractArray, src::AbstractSparseArray)
  SparseArrayInterface.sparse_copyto!(dest, src)
  return dest
end

# Fix ambiguity error
function Base.copyto!(dest::LayoutArray, src::AbstractSparseArray)
  SparseArrayInterface.sparse_copyto!(dest, src)
  return dest
end

function Base.permutedims!(dest::AbstractArray, src::AbstractSparseArray, perm)
  SparseArrayInterface.sparse_permutedims!(dest, src, perm)
  return dest
end

function Base.mapreduce(f, op, a::AbstractSparseArray; kwargs...)
  return SparseArrayInterface.sparse_mapreduce(f, op, a; kwargs...)
end

# TODO: Why isn't this calling `mapreduce` already?
function Base.iszero(a::AbstractSparseArray)
  return SparseArrayInterface.sparse_iszero(a)
end

# TODO: Why isn't this calling `mapreduce` already?
function Base.isreal(a::AbstractSparseArray)
  return SparseArrayInterface.sparse_isreal(a)
end
