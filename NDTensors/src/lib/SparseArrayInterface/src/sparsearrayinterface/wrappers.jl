perm(::PermutedDimsArray{<:Any,<:Any,P}) where {P} = P
genperm(v, perm) = map(j -> v[j], perm)
genperm(v::CartesianIndex, perm) = CartesianIndex(map(j -> Tuple(v)[j], perm))

# TODO: Should the keys get permuted?
sparse_storage(a::PermutedDimsArray) = sparse_storage(parent(a))

function storage_index_to_index(a::PermutedDimsArray, I)
  return genperm(storage_index_to_index(parent(a), I), perm(a))
end

function index_to_storage_index(
  a::PermutedDimsArray{<:Any,N}, I::CartesianIndex{N}
) where {N}
  return storage_index_to_index(parent(a), genperm(I, perm(a)))
end

# TODO: Add `SubArray`, `ReshapedArray`, `Diagonal`, etc.
