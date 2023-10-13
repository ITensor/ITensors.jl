# NDTensors.permutedims
function permutedims(::Type{<:Array}, M, perm)
  return @strided Mdest = Base.permutedims(M, perm)
end

# NDTensors.permutedims!
function permutedims!(::Type{<:Array}, Mdest, ::Type{<:Array}, M, perm)
  return @strided Mdest .= Base.permutedims(M, perm)
end

function permutedims!!(::Type{<:Array}, B, ::Type{<:Array}, A, perm, f)
  @strided B .= f.(B, Base.permutedims(A, perm))
end
