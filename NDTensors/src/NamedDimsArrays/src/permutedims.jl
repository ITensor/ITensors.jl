function Base.permutedims(na::AbstractNamedDimsArray, perm)
  names = map(j -> dimnames(na)[j], perm)
  return align(na, names)
end
