## NOTICE!!: Here we are not importing Base.permutedims or Base.permutedims! but
## are writing our own implementation. This allows us to better dispatch on wrapped arrays

# NDTensors.permutedims!
# function permutedims!(Mdest::AbstractArray, M::AbstractArray, perm)
#   permutedims!(leaf_parenttype(Mdest), Mdest, leaf_parenttype(M), M, perm)
#   return Mdest
# end

# # NDTensors.permutedims!
# function permutedims!(::Type{<:AbstractArray}, Mdest, ::Type{<:AbstractArray}, M, perm)
#   Base.permutedims!(Mdest, M, perm)
#   return Mdest
# end

function permutedims!!(B::AbstractArray, A::AbstractArray, perm)
  return permutedims!!(leaf_parenttype(B), B, leaf_parenttype(A), A, perm)
end

function permutedims!!(
  Bleaftype::Type{<:AbstractArray}, B, Aleaftype::Type{<:AbstractArray}, A, perm
)
  Base.permutedims!(expose(B), expose(A), perm)
  return B
end

function permutedims!!(B::AbstractArray, A::AbstractArray, perm, f)
  return permutedims!!(leaf_parenttype(B), B, leaf_parenttype(A), A, perm, f)
end

function permutedims!!(
  Bleaftype::Type{<:AbstractArray}, B, Aleaftype::Type{<:AbstractArray}, A, perm, f
)
  Base.permutedims!(expose(B), expose(A), perm, f)
  return B
end

# function permutedims!(::Type{<:AbstractArray}, B, ::Type{<:AbstractArray}, A, perm, f)
#   B .= f.(B, Base.permutedims(A, perm))
#   return B
# end
