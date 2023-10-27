function Base.permutedims!(
  Edest::Exposed{<:MtlArray,<:Base.ReshapedArray}, Esrc::Exposed{<:MtlArray}, perm
)
  Aperm = permutedims(Esrc, perm)
  copyto!(expose(parent(Edest)), expose(Aperm))
  return unexpose(Edest)
end

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
