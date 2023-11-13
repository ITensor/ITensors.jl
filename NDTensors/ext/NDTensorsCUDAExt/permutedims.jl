function Base.permutedims!(
    Edest::Exposed{<:CuArray,<:Base.ReshapedArray}, Esrc::Exposed{<:CuArray}, perm
  )
    Aperm = permutedims(Esrc, perm)
    copyto!(expose(parent(Edest)), expose(Aperm))
    return unexpose(Edest)
  end