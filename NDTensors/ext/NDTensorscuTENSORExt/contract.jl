using Base: ReshapedArray
using NDTensors.Expose: expose
using NDTensors: NDTensors, ContractAlgorithm, Dense, DenseTensor, NativeContract, array,
    contract!, default_contract_algorithm, is_applicable
using cuTENSOR: cuTENSOR, CuArray, CuTensor

"""
    cuTENSORDense <: NDTensors.ContractAlgorithm

Algorithm tag for cuTENSOR's dense contraction path. Applies to two
`DenseTensor`s whose backing data type is a `CuArray`. Set as the
default for that input shape when this extension is loaded (via the
`default_contract_algorithm` overload below), so dense CUDA contractions
automatically use cuTENSOR.
"""
struct cuTENSORDense <: ContractAlgorithm end

NDTensors.is_applicable(::cuTENSORDense, ::Type, ::Type) = false
function NDTensors.is_applicable(
        ::cuTENSORDense,
        T1::Type{<:DenseTensor{<:Any, <:Any, <:Dense{<:Any, <:CuArray}}},
        T2::Type{<:DenseTensor{<:Any, <:Any, <:Dense{<:Any, <:CuArray}}}
    )
    return true
end

# Loading this extension makes `cuTENSORDense` the default for dense CUDA
# contractions (matching the behavior of the previous
# `Exposed{<:CuArray, <:DenseTensor}` direct-dispatch method).
function NDTensors.default_contract_algorithm(
        ::Type{<:DenseTensor{<:Any, <:Any, <:Dense{<:Any, <:CuArray}}},
        ::Type{<:DenseTensor{<:Any, <:Any, <:Dense{<:Any, <:CuArray}}}
    )
    return cuTENSORDense()
end

# Handle CuArrays cuTENSOR.jl can't accept directly (non-zero offsets,
# reshaped views).
to_zero_offset_cuarray(a::CuArray) = iszero(a.offset) ? a : copy(a)
to_zero_offset_cuarray(a::ReshapedArray) = copy(expose(a))

function NDTensors.contract!(
        ::cuTENSORDense,
        R::DenseTensor,
        labelsR,
        T1::DenseTensor,
        labelsT1,
        T2::DenseTensor,
        labelsT2,
        α::Number = one(eltype(R)),
        β::Number = zero(eltype(R))
    )
    zoffR = iszero(array(R).offset)
    arrayR = zoffR ? array(R) : copy(array(R))
    arrayT1 = to_zero_offset_cuarray(array(T1))
    arrayT2 = to_zero_offset_cuarray(array(T2))
    # Promote inputs to a common type. cuTENSOR contraction only performs
    # limited promotions of input element types, see e.g.
    # https://github.com/JuliaGPU/CUDA.jl/blob/v5.4.2/lib/cutensor/src/types.jl#L11-L19
    elt = promote_type(eltype.((arrayR, arrayT1, arrayT2))...)
    if elt !== eltype(arrayR)
        return error(
            "In cuTENSOR contraction, input tensors have element types `$(eltype(arrayT1))` and `$(eltype(arrayT2))` while the output has element type `$(eltype(arrayR))`."
        )
    end
    arrayT1 = convert(CuArray{elt}, arrayT1)
    arrayT2 = convert(CuArray{elt}, arrayT2)
    cuR = CuTensor(arrayR, collect(labelsR))
    cuT1 = CuTensor(arrayT1, collect(labelsT1))
    cuT2 = CuTensor(arrayT2, collect(labelsT2))
    try
        cuTENSOR.mul!(cuR, cuT1, cuT2, α, β)
    catch e
        e isa cuTENSOR.CUTENSORError || rethrow()
        # cuTENSOR couldn't run this contraction (typically an unsupported
        # operation). Surface what was suppressed, then fall back to the
        # native (cuBLAS-loop) path. Non-CUTENSORError exceptions (OOM,
        # driver mismatch, version regression, internal bugs) propagate.
        @warn "cuTENSOR dense contract failed; falling back to NativeContract." exception =
            (
            e,
            catch_backtrace(),
        )
        contract!(NativeContract(), R, labelsR, T1, labelsT1, T2, labelsT2, α, β)
        return R
    end
    if !zoffR
        array(R) .= cuR.data
    end
    return R
end
