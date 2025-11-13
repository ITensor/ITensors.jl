parent(E::Exposed) = parent(unexpose(E))

transpose(E::Exposed) = transpose(unexpose(E))

adjoint(E::Exposed) = adjoint(unexpose(E))
getindex(E::Exposed) = unexpose(E)[]

function setindex!(E::Exposed, x::Number)
    unexpose(E)[] = x
    return unexpose(E)
end

getindex(E::Exposed, I...) = unexpose(E)[I...]

function copy(E::Exposed)
    return copy(unexpose(E))
end

any(f, E::Exposed) = any(f, unexpose(E))

print_array(io::IO, E::Exposed) = print_array(io, unexpose(E))
