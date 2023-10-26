## Create the Exposed version of Base.permutedims
## This will be permutedims when NDTensors imports base.permutedims
function Base.permutedims(E::Unwrap.Exposed{<:Array}, perm)
  ## Creating Mperm here to evaluate the permutation and
  ## avoid returning a Stridedview
  @strided Mperm = Base.permutedims(E.object, perm)
  return Mperm
end
# NDTensors.permutedims
function permutedims(::Type{<:Array}, M, perm)
  ## Creating Mperm here to evaluate the permutation and
  ## avoid returning a Stridedview
  @strided Mperm = Base.permutedims(M, perm)
  return Mperm
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
