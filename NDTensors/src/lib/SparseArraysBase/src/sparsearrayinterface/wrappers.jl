using ..NestedPermutedDimsArrays: NestedPermutedDimsArray

## PermutedDimsArray

const AnyPermutedDimsArray{T,N,perm,iperm,P} = Union{
  PermutedDimsArray{T,N,perm,iperm,P},NestedPermutedDimsArray{T,N,perm,iperm,P}
}

# TODO: Use `TypeParameterAccessors`.
perm(::AnyPermutedDimsArray{<:Any,<:Any,Perm}) where {Perm} = Perm
iperm(::AnyPermutedDimsArray{<:Any,<:Any,<:Any,IPerm}) where {IPerm} = IPerm

# TODO: Use `Base.PermutedDimsArrays.genperm` or
# https://github.com/jipolanco/StaticPermutations.jl?
genperm(v, perm) = map(j -> v[j], perm)
genperm(v::CartesianIndex, perm) = CartesianIndex(map(j -> Tuple(v)[j], perm))

function storage_index_to_index(a::AnyPermutedDimsArray, I)
  return genperm(storage_index_to_index(parent(a), I), perm(a))
end

function index_to_storage_index(
  a::AnyPermutedDimsArray{<:Any,N}, I::CartesianIndex{N}
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

function storage_index_to_index(a::SubArray, I)
  return storage_index_to_index(parent(a), I)
end

function index_to_storage_index(a::SubArray{<:Any,N}, I::CartesianIndex{N}) where {N}
  return index_to_storage_index(parent(a), I)
end

getindex_zero_function(a::SubArray) = getindex_zero_function(parent(a))
