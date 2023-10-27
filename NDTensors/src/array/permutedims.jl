## Create the Exposed version of Base.permutedims
## This will be permutedims when NDTensors imports base.permutedims
function Base.permutedims(E::Exposed{<:Array}, perm)
  ## Creating Mperm here to evaluate the permutation and
  ## avoid returning a Stridedview
  @strided Mperm = Base.permutedims(unexpose(E), perm)
  return Mperm
end

function Base.permutedims!(Edest::Exposed{<:Array}, Esrc::Exposed{<:Array}, perm)
  @strided unexpose(Edest) .= Base.permutedims(Esrc, perm)
  return unexpose(Edest)
end

function Base.permutedims!(Edest::Exposed{<:Array}, Esrc::Exposed{<:Array}, perm, f)
  @strided unexpose(Edest) .= f.(unexpose(Edest), Base.permutedims(Esrc, perm))
  return unexpose(Edest)
end

# NDTensors.permutedims!
function permutedims!(::Type{<:Array}, Mdest, ::Type{<:Array}, M, perm)
  @strided Mdest .= Base.permutedims(M, perm)
  return Mdest
end

function permutedims!(::Type{<:Array}, B, ::Type{<:Array}, A, perm, f)
  @strided B .= f.(B, Base.permutedims(A, perm))
  return B
end
