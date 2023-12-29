perm(::PermutedDimsArray{<:Any,<:Any,Perm}) where {Perm} = Perm
iperm(::PermutedDimsArray{<:Any,<:Any,<:Any,IPerm}) where {IPerm} = IPerm

function nonzero_keys(a::PermutedDimsArray)
  return (
    CartesianIndex(Base.PermutedDimsArrays.genperm(Tuple(parent_index), perm(a))) for
    parent_index in nonzero_keys(parent(a))
  )
end
