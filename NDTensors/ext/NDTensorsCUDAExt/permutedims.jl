function NDTensors.permutedims!!(::Type{<:CuArray}, M, perm)
  return permutedims(M, perm)
end
