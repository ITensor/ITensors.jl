using CUDA: CuArray
using LinearAlgebra: LinearAlgebra, mul!, transpose
using NDTensors.Expose: Exposed, expose, unexpose

# This was calling generic matrix multiplication.
# TODO: Raise an issue with `CUDA.jl`.
function LinearAlgebra.mul!(
        CM::Exposed{<:CuArray, <:LinearAlgebra.Transpose},
        AM::Exposed{<:CuArray},
        BM::Exposed{<:CuArray},
        α,
        β,
    )
    mul!(transpose(CM), transpose(BM), transpose(AM), α, β)
    return unexpose(CM)
end

# This was calling generic matrix multiplication.
# TODO: Raise an issue with `CUDA.jl`.
function LinearAlgebra.mul!(
        CM::Exposed{<:CuArray, <:LinearAlgebra.Adjoint},
        AM::Exposed{<:CuArray},
        BM::Exposed{<:CuArray},
        α,
        β,
    )
    mul!(CM', BM', AM', α, β)
    return unexpose(CM)
end

## Fix issue in CUDA.jl where it cannot distinguish Transpose{Reshape{Adjoint{CuArray}}}
## as a CuArray and calls generic matmul
function LinearAlgebra.mul!(
        CM::Exposed{<:CuArray},
        AM::Exposed{<:CuArray},
        BM::Exposed{
            <:CuArray,
            <:LinearAlgebra.Transpose{
                <:Any, <:Base.ReshapedArray{<:Any, <:Any, <:LinearAlgebra.Adjoint},
            },
        },
        α,
        β,
    )
    mul!(CM, AM, expose(transpose(copy(expose(parent(BM))))), α, β)
    return unexpose(CM)
end
