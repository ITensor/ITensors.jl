function permutedims!!(B::AbstractArray, A::AbstractArray, perm)
    permutedims!(expose(B), expose(A), perm)
    return B
end

function permutedims!!(B::AbstractArray, A::AbstractArray, perm, f)
    permutedims!(expose(B), expose(A), perm, f)
    return B
end
