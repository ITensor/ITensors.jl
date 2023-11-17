function Base.permutedims(E::Exposed{<:MtlArray,<:Base.ReshapedArray}, perm)
  A = copy(E)
  return permutedims(expose(A), perm)
end
## Theres an issue in metal that `ReshapedArray' wrapped arrays cannot be permuted
function Base.permutedims!(
  Edest::Exposed{<:MtlArray,<:Base.ReshapedArray}, Esrc::Exposed{<:MtlArray}, perm
)
  Aperm = permutedims(Esrc, perm)
  copyto!(expose(parent(Edest)), expose(Aperm))
  return unexpose(Edest)
end

## Theres an issue in metal that `ReshapedArray' wrapped arrays cannot be permuted
## To get around this copy and permute Esrc, reshape to the size of Edest's parent
## and broadcast into the parent.
function Base.permutedims!(
  Edest::Exposed{<:MtlArray,<:Base.ReshapedArray},
  Esrc::Exposed{<:MtlArray,<:Base.ReshapedArray},
  perm,
  f,
)
  Aperm = unwrap_type(Esrc)(reshape(permutedims(Esrc, perm), size(parent(Edest))))
  parent(Edest) .= f.(parent(Edest), Aperm)
  return unexpose(Edest)
end
