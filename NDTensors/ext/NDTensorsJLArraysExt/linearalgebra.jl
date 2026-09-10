using Adapt: adapt
using JLArrays: JLArray, JLMatrix
using LinearAlgebra: LinearAlgebra, Hermitian, Symmetric, eigen, qr, svd
using NDTensors.Expose: Expose, expose, ql, ql_positive, qr, qr_positive
using NDTensors.GPUArraysCoreExtensions: cpu
using NDTensors: NDTensors
using TypeParameterAccessors: unwrap_array_type

## TODO these should work using a JLArray but there is an error converting the Q from its packed QR form
## back into a JLArray see https://github.com/JuliaGPU/GPUArrays.jl/issues/545. To fix call cpu for now
function Expose.qr(A::Exposed{<:JLArray})
    Q, L = qr(expose(cpu(A)))
    return adapt(unwrap_array_type(A), Matrix(Q)), adapt(unwrap_array_type(A), L)
end
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

## Julia 1.13's `svd!` counts the singular values above a tolerance, which indexes them
## one at a time and so is disallowed on a GPU array. To fix call cpu for now.
function LinearAlgebra.svd(A::Exposed{<:JLMatrix}; kwargs...)
    U, S, V = svd(expose(cpu(A)); kwargs...)
    return adapt.(JLArray, (U, S, V))
end
