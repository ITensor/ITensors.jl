## NOTICE!!: Here we are not importing Base.permutedims or Base.permutedims! but
## are writing our own implementation. This allows us to better dispatch on wrapped arrays

function permutedims!!(B::AbstractArray, A::AbstractArray, perm)
  Base.permutedims!(expose(B), expose(A), perm)
  return B
end

function permutedims!!(B::AbstractArray, A::AbstractArray, perm, f)
  Base.permutedims!(expose(B), expose(A), perm, f)
  return B
end
