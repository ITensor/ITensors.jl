function permutedims(E::Exposed, perm)
  return permutedims(E.object, perm)
end

function permutedims!(Edest::Exposed, Esrc::Exposed, perm)
  return permutedims!(Edest.object, Esrc.object, perm)
end

function permutedims!(Edest::Exposed, Esrc::Exposed, perm, f)
  return Edest.object .= f.(Edest.object, Base.permutedims(Esrc, perm))
end
