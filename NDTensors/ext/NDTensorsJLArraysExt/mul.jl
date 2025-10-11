using JLArrays: JLArray
using LinearAlgebra: LinearAlgebra, mul!, transpose
using NDTensors.Expose: Exposed, expose, unexpose

function LinearAlgebra.mul!(
        CM::Exposed{<:JLArray, <:LinearAlgebra.Transpose},
        AM::Exposed{<:JLArray},
        BM::Exposed{<:JLArray},
        α,
        β,
    )
    mul!(transpose(CM), transpose(BM), transpose(AM), α, β)
    return unexpose(CM)
end

function LinearAlgebra.mul!(
        CM::Exposed{<:JLArray, <:LinearAlgebra.Adjoint},
        AM::Exposed{<:JLArray},
        BM::Exposed{<:JLArray},
        α,
        β,
    )
    mul!(CM', BM', AM', α, β)
    return unexpose(CM)
end

## Fix issue in JLArrays.jl where it cannot distinguish Transpose{Reshape{Adjoint{JLArray}}}
## as a JLArray and calls generic matmul
function LinearAlgebra.mul!(
        CM::Exposed{<:JLArray},
        AM::Exposed{<:JLArray},
        BM::Exposed{
            <:JLArray,
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
