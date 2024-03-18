using NDTensors.AMDGPUExtensions: roc
using NDTensors.Expose: Exposed
using NDTensors.GPUArraysCoreExtensions: cpu
using LinearAlgebra: svd

function LinearAlgebra.svd(A::Exposed{<:ROCMatrix}; kwargs...)
  U, S, V = svd(cpu(A))
  return roc.((U, S, V))
end
