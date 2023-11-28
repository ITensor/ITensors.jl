perm(::PermutedDimsArray{<:Any,<:Any,P}) where {P} = P
genperm(v, perm) = map(j -> v[j], perm)
genperm(v::CartesianIndex, perm) = CartesianIndex(map(j -> Tuple(v)[j], perm))

nonzeros(a::PermutedDimsArray) = nonzeros(parent(a))
function nonzero_index_to_index(a::PermutedDimsArray, Inz)
  return genperm(nonzero_index_to_index(parent(a), Inz), perm(a))
end
function index_to_nonzero_index(
  a::PermutedDimsArray{<:Any,N}, I::CartesianIndex{N}
) where {N}
  return nonzero_index_to_index(parent(a), genperm(I, perm(a)))
end

# TODO: Add `SubArray`, `ReshapedArray`, `Diagonal`, etc.
