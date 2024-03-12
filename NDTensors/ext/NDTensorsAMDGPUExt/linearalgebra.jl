function LinearAlgebra.svd(A::Exposed{<:ROCMatrix}; kwargs...)
  U, S, V = svd(NDTensors.cpu(A))
  return NDTensors.roc.((U, S, V))
end
