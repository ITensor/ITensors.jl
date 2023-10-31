parent(E::Exposed) = parent(unexpose(E))

transpose(E::Exposed) = transpose(unexpose(E))

cpu(E::Exposed) = cpu(unexpose(E))

getindex(E::Exposed) = getindex(unexpose(E))

function setindex!(E::Exposed, x::Number)
  setindex!(unexpose(E), x)
  return unexpose(E)
end