using Base: ReshapedArray
using NDTensors.Expose: Exposed, expose, unexpose
using NDTensors: NDTensors, BlockSparseTensor, DenseTensor, array,
blockdims, data, eachnzblock, inds, nblocks, nzblocks
using cuTENSOR: cuTENSOR, CuArray, CuTensor

# Handle cases that can't be handled by `cuTENSOR.jl`
# right now.
function to_zero_offset_cuarray(a::CuArray)
    return iszero(a.offset) ? a : copy(a)
end
function to_zero_offset_cuarray(a::ReshapedArray)
    return copy(expose(a))
end

function block_extents(ind)
    return ntuple(i -> ind.space[i].second, nblocks(ind))
end

#### Functions to turn Tensors into BlockSparseCuTensors for contraction
function to_cuTensorBS(T::BlockSparseTensor)
    blocks_t1 = []
    # T = tensor(target)
    for blockT in eachnzblock(T)
        offsetT = NDTensors.offset(T, blockT)
        blockdimsT = blockdims(T, blockT)
        blockdimT = prod(blockdimsT)
        push!(blocks_t1, @view data(T)[(offsetT + 1):(offsetT + blockdimT)])
    end
    blocks_t1 = Vector{typeof(blocks_t1[1])}(blocks_t1)
    block_extents_t1 = [block_extents(idx) for idx in inds(T)] ## This is sections
    nzblock_coords_t1 = [Int64.(x.data) for x in nzblocks(T)]
    block_per_mode_t1 = length.(block_extents_t1)
    is = [i for i in 1:ndims(T)]
    return cuTENSOR.CuTensorBS(blocks_t1, block_per_mode_t1, block_extents_t1, nzblock_coords_t1, is);
end

function NDTensors._contract!(R::Exposed{<:CuArray, <:BlockSparseTensor},
        labelsR,
        tensor1::Exposed{<:CuArray, <:BlockSparseTensor},
        labelstensor1,
        tensor2::Exposed{<:CuArray, <:BlockSparseTensor},
        labelstensor2,
        grouped_contraction_plan,
        executor,
    )
    N1 = ndims(unexpose(tensor1)) 
    N2 = ndims(unexpose(tensor2)) 
    NR = ndims(unexpose(R)) 
    if NDTensors.using_CuTensorBS() && (N1 > 0) && (N2 > 0) && (NR > 0)
        # println("Using new function")
        cuR = ITensor_to_cuTensorBS(unexpose(R))
        cutensor1 = ITensor_to_cuTensorBS(unexpose(tensor1))
        cutensor2 = ITensor_to_cuTensorBS(unexpose(tensor2))

        cuR.inds = [labelsR...]
        cutensor1.inds = [labelstensor1...]
        cutensor2.inds = [labelstensor2...]

        cuTENSOR.mul!(cuR, cutensor1, cutensor2, 1.0, 0.0)
        return R
    else
        return NDTensors._contract!(
        unexpose(R),
        labelsR,
        unexpose(tensor1),
        labelstensor1,
        unexpose(tensor2),
        labelstensor2,
        grouped_contraction_plan,
        executor,
        )
    end
end

function NDTensors.contract!(
        exposedR::Exposed{<:CuArray, <:DenseTensor},
        labelsR,
        exposedT1::Exposed{<:CuArray, <:DenseTensor},
        labelsT1,
        exposedT2::Exposed{<:CuArray, <:DenseTensor},
        labelsT2,
        α::Number = one(Bool),
        β::Number = zero(Bool)
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
        # Fall back to default contraction (cuBLAS) for operations
        # cuTENSOR doesn't support.
        NDTensors.contract!(R, labelsR, T1, labelsT1, T2, labelsT2, α, β)
        return R
    end
    if !zoffR
        array(R) .= cuR.data
    end
    return R
end
