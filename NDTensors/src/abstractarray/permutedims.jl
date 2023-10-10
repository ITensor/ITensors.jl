function permutedims!!(::Type{<:AbstractArray}, M, perm)
  return @strided Mdest = permutedims(M, perm)
end
