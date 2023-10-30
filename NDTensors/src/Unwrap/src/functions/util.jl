Base.parent(E::Exposed) = parent(unexpose(E))

Base.transpose(E::Expose) = transpose(unexpose(E))