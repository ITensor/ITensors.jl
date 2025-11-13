using Adapt: adapt
using JLArrays: JLArray, JLMatrix
using LinearAlgebra: LinearAlgebra, Hermitian, Symmetric, qr, eigen
using NDTensors: NDTensors
using NDTensors.Expose: Expose, expose, qr, qr_positive, ql, ql_positive
using NDTensors.GPUArraysCoreExtensions: cpu
using NDTensors.Vendored.TypeParameterAccessors: unwrap_array_type

## TODO this function exists because of the same issue below. when
## that issue is resolved we can rely on the abstractarray version of
## this operation.
function Expose.qr(A::Exposed{<:JLArray})
    Q, L = qr(unexpose(A))
    return adapt(unwrap_array_type(A), Matrix(Q)), adapt(unwrap_array_type(A), L)
end
## TODO this should work using a JLArray but there is an error converting the Q from its packed QR from
## back into a JLArray see https://github.com/JuliaGPU/GPUArrays.jl/issues/545. To fix call cpu for now
function Expose.qr_positive(A::Exposed{<:JLArray})
    Q, L = qr_positive(expose(cpu(A)))
    return adapt(unwrap_array_type(A), copy(Q)), adapt(unwrap_array_type(A), L)
end

function Expose.ql(A::Exposed{<:JLMatrix})
    Q, L = ql(expose(cpu(A)))
    return adapt(unwrap_array_type(A), copy(Q)), adapt(unwrap_array_type(A), L)
end
function Expose.ql_positive(A::Exposed{<:JLMatrix})
    Q, L = ql_positive(expose(cpu(A)))
    return adapt(unwrap_array_type(A), copy(Q)), adapt(unwrap_array_type(A), L)
end

function LinearAlgebra.eigen(A::Exposed{<:JLMatrix, <:Symmetric})
    q, l = (eigen(expose(cpu(A))))
    return adapt.(unwrap_array_type(A), (q, l))
end

function LinearAlgebra.eigen(A::Exposed{<:JLMatrix, <:Hermitian})
    q, l = (eigen(expose(Hermitian(cpu(unexpose(A).data)))))
    return adapt.(JLArray, (q, l))
end
