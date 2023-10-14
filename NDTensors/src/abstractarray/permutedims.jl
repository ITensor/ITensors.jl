## NOTICE!!: Here we are not importing Base.permutedims or Base.permutedims! but
## are writing our own implementation. This allows us to 
# NDTensors.permutedims
function permutedims(M::AbstractArray, perm)
  return permutedims(leaf_parenttype(M), M, perm)
end

# NDTensors.permutedims
function permutedims(::Type{<:AbstractArray}, M, perm)
  return Base.permutedims(M, perm)
end

# NDTensors.permutedims!
function permutedims!(Mdest::AbstractArray, M::AbstractArray, perm)
  return permutedims!(leaf_parenttype(Mdest), Mdest, leaf_parenttype(M), M, perm)
end

# NDTensors.permutedims!
function permutedims!(::Type{<:AbstractArray}, Mdest, ::Type{<:AbstractArray}, M, perm)
  return Base.permutedims!(Mdest, M, perm)
end

function permutedims!!(B::AbstractArray, A::AbstractArray, perm, f)
  return permutedims!!(leaf_parenttype(B), B, leaf_parenttype(A), A, perm, f)
end

function permutedims!!(Bleaftype::Type{<:AbstractArray}, B, Aleaftype::Type{<:AbstractArray}, A, perm, f)
  permutedims!(Bleaftype, B, Aleaftype, A, perm, f)
  return B
end

function permutedims!(::Type{<:AbstractArray}, B, ::Type{<:AbstractArray}, A, perm, f)
  B .= f.(B, Base.permutedims(A, perm))
  return B
end
