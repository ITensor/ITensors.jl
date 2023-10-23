function NDTensors.permutedims!(
  ::Type{<:MtlArray},
  Adest::Base.ReshapedArray{<:Any,<:Any,<:SubArray},
  ::Type{<:MtlArray},
  A,
  perm,
)
  Aperm = permutedims(A, perm)
  Adest_parent = parent(Adest)
  copyto!(Adest_parent, Aperm)
  return Adest
end
