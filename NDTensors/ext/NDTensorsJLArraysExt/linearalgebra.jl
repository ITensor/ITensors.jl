using Adapt: adapt
using JLArrays: JLMatrix
using NDTensors: NDTensors
using NDTensors.Expose: Expose, expose, ql, ql_positive
using NDTensors.GPUArraysCoreExtensions: cpu
using NDTensors.TypeParameterAccessors: unwrap_array_type

## TODO currently AMDGPU doesn't have ql so make a ql function
function Expose.ql(A::Exposed{<:JLMatrix})
  Q, L = ql(expose(cpu(A)))
  return adapt(unwrap_array_type(A), Matrix(Q)), adapt(unwrap_array_type(A), L)
end
function Expose.ql_positive(A::Exposed{<:JLMatrix})
  Q, L = ql_positive(expose(cpu(A)))
  return adapt(unwrap_array_type(A), Matrix(Q)), adapt(unwrap_array_type(A), L)
end
