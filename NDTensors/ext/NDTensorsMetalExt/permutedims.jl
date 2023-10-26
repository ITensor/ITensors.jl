function Base.permutedims!(Edest::Unwrap.Exposed{<:MtlArray}, Esrc::Unwraped.Exposed{<:MtlArray}, perm)
  Eperm = expose(permutedims(Esrc), perm)
  copyto!(parent(Edest.object), Eperm.object)
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
