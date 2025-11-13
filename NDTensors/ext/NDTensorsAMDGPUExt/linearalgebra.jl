using NDTensors.AMDGPUExtensions: roc
using NDTensors.Expose: Expose, Exposed, expose, ql, ql_positive
using NDTensors.GPUArraysCoreExtensions: cpu
using NDTensors.Vendored.TypeParameterAccessors: unwrap_array_type
using LinearAlgebra: svd
using Adapt: adapt
using AMDGPU: ROCMatrix

function LinearAlgebra.svd(A::Exposed{<:ROCMatrix}; kwargs...)
    U, S, V = svd(cpu(A))
    return roc.((U, S, V))
end

## TODO currently AMDGPU doesn't have ql so make a ql function
function Expose.ql(A::Exposed{<:ROCMatrix})
    Q, L = ql(expose(cpu(A)))
    return adapt(unwrap_array_type(A), Matrix(Q)), adapt(unwrap_array_type(A), L)
end
function Expose.ql_positive(A::Exposed{<:ROCMatrix})
    Q, L = ql_positive(expose(cpu(A)))
    return adapt(unwrap_array_type(A), Matrix(Q)), adapt(unwrap_array_type(A), L)
end
