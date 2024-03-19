using NDTensors.AMDGPUExtensions: roc
using NDTensors.Expose: Exposed, ql, ql_positive
using NDTensors.GPUArraysCoreExtensions: cpu
using NDTensors.TypeParameterAccessors: unwrap_array_type
using LinearAlgebra: svd

function LinearAlgebra.svd(A::Exposed{<:ROCMatrix}; kwargs...)
  U, S, V = svd(cpu(A))
  return roc.((U, S, V))
end

## TODO currently AMDGPU doesn't have ql so make a ql function
function NDTensors.Expose.ql(A::Exposed{<:ROCMatrix})
  Q, L = ql(expose(NDTensors.cpu(A)))
  return adapt(unwrap_array_type(A), Matrix(Q)), adapt(unwrap_array_type(A), L)
end
function NDTensors.Expose.ql_positive(A::Exposed{<:ROCMatrix})
  Q, L = ql_positive(expose(NDTensors.cpu(A)))
  return adapt(unwrap_array_type(A), Matrix(Q)), adapt(unwrap_array_type(A), L)
end