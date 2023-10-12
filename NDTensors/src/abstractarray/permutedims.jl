function NDTensors.permutedims(::Type{<:AbstractArray}, M, perm)
  return Base.permutedims(M, perm)
end

function permutedims!(::Type{<:AbstractArray}, Mdest, ::Type{<:AbstractArray}, M, perm)
  return Mdest .= Base.permutedims(M, perm)
end

function permutedims!!(B::AbstractArray, A::AbstractArray, perm, f)
  return permutedims!!(leaf_parenttype(B), B, leaf_parenttype(A), A, perm, f)
end

function permutedims!!(::Type{<:AbstractArray}, B, ::Type{<:AbstractArray}, A, perm, f)
  return B .= f.(B, Base.permutedims(A, perm))
end
