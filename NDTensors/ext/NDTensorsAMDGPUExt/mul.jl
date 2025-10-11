using NDTensors.Expose: Exposed, expose, parent, unexpose
using LinearAlgebra: LinearAlgebra, Adjoint, Transpose, mul!
using AMDGPU: ROCArray

# This was calling generic matrix multiplication.
function LinearAlgebra.mul!(
        CM::Exposed{<:ROCArray, <:LinearAlgebra.Transpose},
        AM::Exposed{<:ROCArray},
        BM::Exposed{<:ROCArray},
        α,
        β,
    )
    mul!(transpose(CM), transpose(BM), transpose(AM), α, β)
    return unexpose(CM)
end

# This was calling generic matrix multiplication.
function LinearAlgebra.mul!(
        CM::Exposed{<:ROCArray, <:LinearAlgebra.Adjoint},
        AM::Exposed{<:ROCArray},
        BM::Exposed{<:ROCArray},
        α,
        β,
    )
    mul!(CM', BM', AM', α, β)
    return unexpose(CM)
end

# Fix issue in AMDGPU.jl where it cannot distinguish
# Transpose{Reshape{Adjoint{ROCArray}}} as a ROCArray and calls generic matmul
function LinearAlgebra.mul!(
        CM::Exposed{<:ROCArray},
        AM::Exposed{<:ROCArray},
        BM::Exposed{
            <:ROCArray,
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
