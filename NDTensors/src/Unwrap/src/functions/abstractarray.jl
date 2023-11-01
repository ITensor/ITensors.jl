parent(E::Exposed) = parent(unexpose(E))

transpose(E::Exposed) = transpose(unexpose(E))

cpu(E::Exposed) = cpu(unexpose(E))

getindex(E::Exposed) = unexpose(E)[]

function setindex!(E::Exposed, x::Number)
  unexpose(E)[] = x
  return unexpose(E)
end

getindex(E::Exposed, I...) = unexpose(E)[I...]

function copy(E::Exposed)
  return copy(unexpose(E))
end
