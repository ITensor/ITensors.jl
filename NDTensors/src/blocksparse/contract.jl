using .BackendSelection: @Algorithm_str, Algorithm

# Determine the contraction output and block contractions. Bundles `R` and
# the contraction plan in a `TensorAndContractionPlan` so the plan can flow
# through the universal `contract!(dest, ...)` entry without changing the
# entry-point signature across tensor types.
function contraction_output(
        tensor1::BlockSparseTensor,
        labelstensor1,
        tensor2::BlockSparseTensor,
        labelstensor2,
        labelsR
    )
    indsR =
        contract_inds(inds(tensor1), labelstensor1, inds(tensor2), labelstensor2, labelsR)
    TensorR = contraction_output_type(typeof(tensor1), typeof(tensor2), indsR)
    blockoffsetsR, contraction_plan = contract_blockoffsets(
        blockoffsets(tensor1),
        inds(tensor1),
        labelstensor1,
        blockoffsets(tensor2),
        inds(tensor2),
        labelstensor2,
        indsR,
        labelsR
    )
    R = similar(TensorR, blockoffsetsR, indsR)
    return TensorAndContractionPlan(R, contraction_plan)
end

function contract_blockoffsets(
        boffs1::BlockOffsets, inds1, labels1, boffs2::BlockOffsets, inds2, labels2, indsR,
        labelsR
    )
    alg = Algorithm"sequential"()
    if using_threaded_blocksparse() && nthreads() > 1
        alg = Algorithm"threaded_threads"()
    end
    return contract_blockoffsets(
        alg, boffs1, inds1, labels1, boffs2, inds2, labels2, indsR, labelsR
    )
end

# `BlockSparseContract` orchestrates the block-pair iteration; per-block
# dense contractions are delegated to `bsc.inner`. Native default for
# `BlockSparseTensor × BlockSparseTensor`.
function is_applicable(
        ::BlockSparseContract,
        ::Type{<:BlockSparseTensor},
        ::Type{<:BlockSparseTensor}
    )
    return true
end

function default_contract_algorithm(
        ::Type{<:BlockSparseTensor},
        ::Type{<:BlockSparseTensor}
    )
    return BlockSparseContract()
end

# `NativeContract` is the per-leaf algorithm in this scaffold; it no
# longer carries the BS orchestration. A `with_contract_algorithm(
# NativeContract())` scope on a BS×BS pair therefore falls through to
# `default_contract_algorithm` (i.e. `BlockSparseContract(NativeContract())`)
# rather than picking `NativeContract()` directly and hitting a method
# error for the absent `contract!(::NativeContract, ::BlockSparseTensor,
# ...)`.
function is_applicable(
        ::NativeContract,
        ::Type{<:BlockSparseTensor},
        ::Type{<:BlockSparseTensor}
    )
    return false
end

function contract!(
        bsc::BlockSparseContract,
        dest::TensorAndContractionPlan{T},
        labelsR,
        tensor1::BlockSparseTensor,
        labelstensor1,
        tensor2::BlockSparseTensor,
        labelstensor2,
        α::Number = one(eltype(dest.tensor)),
        β::Number = zero(eltype(dest.tensor))
    ) where {T <: BlockSparseTensor}
    R = dest.tensor
    contraction_plan = dest.contraction_plan
    isempty(contraction_plan) && return R
    if using_threaded_blocksparse() && nthreads() > 1
        return contract_blocksparse_threaded_folds!(
            bsc.inner,
            R, labelsR, tensor1, labelstensor1, tensor2, labelstensor2, contraction_plan,
            α, β
        )::T
    end
    return contract_blocksparse_sequential!(
        bsc.inner,
        R, labelsR, tensor1, labelstensor1, tensor2, labelstensor2, contraction_plan,
        α, β
    )::T
end

# `mul!`-style entry for `BlockSparseTensor` outputs that arrive without a
# precomputed contraction plan (the plan is normally produced by
# `contraction_output` for the `*` flow). Build the plan from the input
# blocks against `R`'s indices, then delegate to the
# `TensorAndContractionPlan`-keyed method above.
function contract!(
        bsc::BlockSparseContract,
        R::BlockSparseTensor,
        labelsR,
        tensor1::BlockSparseTensor,
        labelstensor1,
        tensor2::BlockSparseTensor,
        labelstensor2,
        α::Number = one(eltype(R)),
        β::Number = zero(eltype(R))
    )
    _, contraction_plan = contract_blockoffsets(
        blockoffsets(tensor1), inds(tensor1), labelstensor1,
        blockoffsets(tensor2), inds(tensor2), labelstensor2,
        inds(R), labelsR
    )
    return contract!(
        bsc,
        TensorAndContractionPlan(R, contraction_plan),
        labelsR, tensor1, labelstensor1, tensor2, labelstensor2,
        α, β
    )
end
