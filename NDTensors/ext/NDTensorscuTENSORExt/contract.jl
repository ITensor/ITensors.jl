using Base: ReshapedArray
using NDTensors: NDTensors, DenseTensor, array
using NDTensors.Expose: Exposed, expose, unexpose
using cuTENSOR: cuTENSOR, CuArray, CuTensor

# Handle cases that can't be handled by `cuTENSOR.jl`
# right now.
function to_zero_offset_cuarray(a::CuArray)
    return iszero(a.offset) ? a : copy(a)
end
function to_zero_offset_cuarray(a::ReshapedArray)
    return copy(expose(a))
end

function NDTensors.contract!(
        exposedR::Exposed{<:CuArray, <:DenseTensor},
        labelsR,
        exposedT1::Exposed{<:CuArray, <:DenseTensor},
        labelsT1,
        exposedT2::Exposed{<:CuArray, <:DenseTensor},
        labelsT2,
        α::Number = one(Bool),
        β::Number = zero(Bool),
    )
    R, T1, T2 = unexpose.((exposedR, exposedT1, exposedT2))
    zoffR = iszero(array(R).offset)
    arrayR = zoffR ? array(R) : copy(array(R))
    arrayT1 = to_zero_offset_cuarray(array(T1))
    arrayT2 = to_zero_offset_cuarray(array(T2))
    # Promote to a common type. This is needed because as of
    # cuTENSOR.jl v5.4.2, cuTENSOR contraction only performs
    # limited sets of type promotions of inputs, see:
    # https://github.com/JuliaGPU/CUDA.jl/blob/v5.4.2/lib/cutensor/src/types.jl#L11-L19
    elt = promote_type(eltype.((arrayR, arrayT1, arrayT2))...)
    if elt !== eltype(arrayR)
        return error(
            "In cuTENSOR contraction, input tensors have element types `$(eltype(arrayT1))` and `$(eltype(arrayT2))` while the output has element type `$(eltype(arrayR))`.",
        )
    end
    arrayT1 = convert(CuArray{elt}, arrayT1)
    arrayT2 = convert(CuArray{elt}, arrayT2)
    cuR = CuTensor(arrayR, collect(labelsR))
    cuT1 = CuTensor(arrayT1, collect(labelsT1))
    cuT2 = CuTensor(arrayT2, collect(labelsT2))
    cuTENSOR.mul!(cuR, cuT1, cuT2, α, β)
    if !zoffR
        array(R) .= cuR.data
    end
    return R
end
