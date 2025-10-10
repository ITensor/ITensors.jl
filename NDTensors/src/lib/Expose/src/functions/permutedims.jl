function permutedims(E::Exposed, perm)
    return permutedims(unexpose(E), perm)
end

function permutedims!(Edest::Exposed, Esrc::Exposed, perm)
    permutedims!(unexpose(Edest), unexpose(Esrc), perm)
    return unexpose(Edest)
end

function permutedims!(Edest::Exposed, Esrc::Exposed, perm, f)
    unexpose(Edest) .= f.(unexpose(Edest), permutedims(Esrc, perm))
    return unexpose(Edest)
end
