# Map
function Base.map!(f, dest::AbstractArray, src::SparseArrayDOK)
  SparseArrayInterface.sparse_map!(f, dest, src)
  return dest
end

function Base.copy!(dest::AbstractArray, src::SparseArrayDOK)
  SparseArrayInterface.sparse_copy!(dest, src)
  return dest
end

function Base.copyto!(dest::AbstractArray, src::SparseArrayDOK)
  SparseArrayInterface.sparse_copyto!(dest, src)
  return dest
end

function Base.permutedims!(dest::AbstractArray, src::SparseArrayDOK, perm)
  SparseArrayInterface.sparse_permutedims!(dest, src, perm)
  return dest
end

function Base.mapreduce(f, op, a::SparseArrayDOK; kwargs...)
  return SparseArrayInterface.sparse_mapreduce(f, op, a; kwargs...)
end

# TODO: Why isn't this calling `mapreduce` already?
function Base.iszero(a::SparseArrayDOK)
  return SparseArrayInterface.sparse_iszero(a)
end

# TODO: Why isn't this calling `mapreduce` already?
function Base.isreal(a::SparseArrayDOK)
  return SparseArrayInterface.sparse_isreal(a)
end
