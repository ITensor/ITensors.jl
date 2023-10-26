function permutedims(E::Exposed{<:AbstractArray}, perm)
  return permutedims(E.object, perm)
end
