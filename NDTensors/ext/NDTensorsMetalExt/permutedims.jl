function Base.permutedims!(Edest::Exposed{<:MtlArray}, Esrc::Exposed{<:MtlArray}, perm)
  Eperm = expose(permutedims(Esrc, perm))
  return copyto!(parent(Edest.object), Eperm.object)
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
