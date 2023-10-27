## Create the Exposed version of Base.permutedims
function permutedims(E::Exposed{<:Array}, perm)
  ## Creating Mperm here to evaluate the permutation and
  ## avoid returning a Stridedview
  @strided Mperm = permutedims(unexpose(E), perm)
  return Mperm
end

function permutedims!(Edest::Exposed{<:Array}, Esrc::Exposed{<:Array}, perm)
  @strided unexpose(Edest) .= permutedims(Esrc, perm)
  return unexpose(Edest)
end

function permutedims!(Edest::Exposed{<:Array}, Esrc::Exposed{<:Array}, perm, f)
  @strided unexpose(Edest) .= f.(unexpose(Edest), permutedims(Esrc, perm))
  return unexpose(Edest)
end
