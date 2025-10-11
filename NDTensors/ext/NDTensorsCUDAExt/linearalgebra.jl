using Adapt: adapt
using CUDA: CUDA, CuMatrix
using LinearAlgebra: Adjoint, svd
using NDTensors: NDTensors
using NDTensors.Expose: Expose, expose, ql, ql_positive
using NDTensors.GPUArraysCoreExtensions: cpu
using NDTensors.Vendored.TypeParameterAccessors: unwrap_array_type
function NDTensors.svd_catch_error(A::CuMatrix; alg::String = "jacobi_algorithm")
    if alg == "jacobi_algorithm"
        alg = CUDA.CUSOLVER.JacobiAlgorithm()
    elseif alg == "qr_algorithm"
        alg = CUDA.CUSOLVER.QRAlgorithm()
    else
        error(
            "svd algorithm $alg is not currently supported. Please see the documentation for currently supported algorithms.",
        )
    end
    return NDTensors.svd_catch_error(A, alg)
end

function NDTensors.svd_catch_error(A::CuMatrix, ::CUDA.CUSOLVER.JacobiAlgorithm)
    USV = try
        svd(A; alg = CUDA.CUSOLVER.JacobiAlgorithm())
    catch
        return nothing
    end
    return USV
end

function NDTensors.svd_catch_error(A::CuMatrix, ::CUDA.CUSOLVER.QRAlgorithm)
    s = size(A)
    if s[1] < s[2]
        At = copy(Adjoint(A))

        USV = try
            svd(At; alg = CUDA.CUSOLVER.QRAlgorithm())
        catch
            return nothing
        end
        MV, MS, MU = USV
        USV = (MU, MS, MV)
    else
        USV = try
            svd(A; alg = CUDA.CUSOLVER.QRAlgorithm())
        catch
            return nothing
        end
    end
    return USV
end

## TODO currently AMDGPU doesn't have ql so make a ql function
function Expose.ql(A::Exposed{<:CuMatrix})
    Q, L = ql(expose(cpu(A)))
    return adapt(unwrap_array_type(A), Matrix(Q)), adapt(unwrap_array_type(A), L)
end
function Expose.ql_positive(A::Exposed{<:CuMatrix})
    Q, L = ql_positive(expose(cpu(A)))
    return adapt(unwrap_array_type(A), Matrix(Q)), adapt(unwrap_array_type(A), L)
end
