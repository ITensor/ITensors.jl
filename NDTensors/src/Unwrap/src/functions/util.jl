Base.parent(E::Exposed) = parent(unexpose(E))

Base.transpose(E::Expose) = transpose(unexpose(E))

NDTensors.cpu(E::Expose) = NDTensors.cpu(unexpose(E))