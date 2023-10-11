function permutedims(::Type{<:Array}, M, perm)
  return @strided Mdest = Base.permutedims(M, perm)
end

function permutedims!(::Type{<:Array}, Mdest, ::Type{<:Array}, M, perm)
  return @strided Mdest .= Base.permutedims(M, perm)
end

function permutedims!!(::Type{<:Array}, B, ::Type{<:Array}, A, perm, f)
  @strided B .= f.(B, Base.permutedims(A, perm))
end

function NDTensors.permutedims(::Type{<:AbstractArray}, M, perm)
  return Base.permutedims(M, perm)
end

function permutedims!(::Type{<:AbstractArray}, Mdest, ::Type{<:AbstractArray}, M, perm)
  return Mdest .= Base.permutedims(M, perm)
end

function permutedims!!(::Type{<:AbstractArray}, B, ::Type{<:AbstractArray}, A, perm, f)
  B .= f.(B, Base.permutedims(A, perm))
end