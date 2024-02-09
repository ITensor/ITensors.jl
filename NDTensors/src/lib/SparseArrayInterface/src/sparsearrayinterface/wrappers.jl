## PermutedDimsArray

perm(::PermutedDimsArray{<:Any,<:Any,P}) where {P} = P
iperm(::PermutedDimsArray{<:Any,<:Any,<:Any,IP}) where {IP} = IP

# TODO: Use `Base.PermutedDimsArrays.genperm` or
# https://github.com/jipolanco/StaticPermutations.jl?
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
  return index_to_storage_index(parent(a), genperm(I, perm(a)))
end

# TODO: Add `getindex_zero_function` definition?

## SubArray

function map_index(
  indices::Tuple{Vararg{Any,N}}, cartesian_index::CartesianIndex{N}
) where {N}
  index = Tuple(cartesian_index)
  new_index = ntuple(length(indices)) do i
    findfirst(==(index[i]), indices[i])
  end
  any(isnothing, new_index) && return nothing
  return CartesianIndex(new_index)
end

# TODO: Check if this is efficient, or determine if this mapping should
# be performed in `storage_index_to_index` and/or `index_to_storage_index`.
function sparse_storage(a::SubArray)
  parent_storage = sparse_storage(parent(a))
  all_sliced_storage_indices = map(keys(parent_storage)) do I
    return map_index(a.indices, I)
  end
  sliced_storage_indices = filter(!isnothing, all_sliced_storage_indices)
  sliced_parent_storage = map(I -> parent_storage[I], keys(sliced_storage_indices))
  return typeof(parent_storage)(sliced_storage_indices, sliced_parent_storage)
end

function storage_index_to_index(a::SubArray, I)
  return storage_index_to_index(parent(a), I)
end

function index_to_storage_index(a::SubArray{<:Any,N}, I::CartesianIndex{N}) where {N}
  return index_to_storage_index(parent(a), I)
end

getindex_zero_function(a::SubArray) = getindex_zero_function(parent(a))
