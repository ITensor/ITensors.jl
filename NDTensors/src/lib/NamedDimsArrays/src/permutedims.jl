function Base.permutedims(na::AbstractNamedDimsArray, perm)
  names = map(j -> dimnames(na)[j], perm)
  return named(permutedims(unname(na), perm), names)
end
