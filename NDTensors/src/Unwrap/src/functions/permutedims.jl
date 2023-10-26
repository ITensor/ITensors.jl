function permutedims(E::Exposed{<:AbstractArray}, perm)
    permutedims(E.object, perm)
end