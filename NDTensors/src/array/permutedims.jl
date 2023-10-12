function permutedims(::Type{<:Array}, M, perm)
  return @strided Mdest = Base.permutedims(M, perm)
end

## TODO is it possible to do something like f = identity and remove the permutedims!! funciton?
function permutedims!(::Type{<:Array}, Mdest, ::Type{<:Array}, M, perm)
  return @strided Mdest .= Base.permutedims(M, perm)
end

function permutedims!!(::Type{<:Array}, B, ::Type{<:Array}, A, perm, f)
  @strided B .= f.(B, Base.permutedims(A, perm))
end