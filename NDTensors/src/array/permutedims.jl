# NDTensors.permutedims
function permutedims(::Type{<:Array}, M, perm)
  return @strided A = Base.permutedims(M, perm)
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
