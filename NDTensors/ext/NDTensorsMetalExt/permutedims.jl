function Base.permutedims!(
  Edest::Exposed{<:MtlArray,<:Base.ReshapedArray}, Esrc::Exposed{<:MtlArray}, perm
)
  Aperm = permutedims(Esrc, perm)
  copyto!(expose(parent(Edest)), expose(Aperm))
  return unexpose(Edest)
end
