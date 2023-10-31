Base.parent(E::Exposed) = parent(unexpose(E))

Base.transpose(E::Exposed) = transpose(unexpose(E))

cpu(E::Exposed) = cpu(unexpose(E))
