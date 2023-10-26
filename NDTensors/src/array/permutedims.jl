## Create the Exposed version of Base.permutedims
## This will be permutedims when NDTensors imports base.permutedims
function Base.permutedims(E::Unwrap.Exposed{<:Array}, perm)
  ## Creating Mperm here to evaluate the permutation and
  ## avoid returning a Stridedview
  @strided Mperm = Base.permutedims(E.object, perm)
  return Mperm
end

function Base.permutedims!(Edest::Unwrap.Exposed{<:Array}, Esrc::Unwrap.Exposed{<:Array}, perm)
  @strided Edest.object .= Base.permutedims(Esrc, perm)
end

function Base.permutedims!(Edest::Unwrap.Exposed{<:Array}, Esrc::Unwrap.Exposed{<:Array}, perm, f)
  @strided Edest.object .= f.(Edest.object, Base.permutedims(Esrc, perm))
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
