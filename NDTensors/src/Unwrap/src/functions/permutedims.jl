function permutedims(E::Exposed, perm)
  return permutedims(E.object, perm)
end

function permutedims!(Edest::Exposed, Esrc::Exposed, perm)
    permutedims!(Edest.object, Esrc.object, perm)
end

function permutedims!(Edest::Exposed, Esrc::Exposed, perm, f)
    Edest.object .= f.(Edest.object, Base.permutedims(Esrc, perm))
end