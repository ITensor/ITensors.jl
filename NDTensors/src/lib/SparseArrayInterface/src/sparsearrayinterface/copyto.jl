function sparse_copy!(dest::AbstractArray, src::AbstractArray)
  @assert axes(dest) == axes(src)
  sparse_map!(identity, dest, src)
  return dest
end

function sparse_copyto!(dest::AbstractArray, src::AbstractArray)
  sparse_map!(identity, dest, src)
  return dest
end

function sparse_permutedims!(dest::AbstractArray, src::AbstractArray, perm)
  sparse_copyto!(dest, PermutedDimsArray(src, perm))
  return dest
end
