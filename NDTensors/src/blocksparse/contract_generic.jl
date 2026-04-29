# A generic version that is used by both
# "threaded_folds" and "threaded"threads".
function contract_blockoffsets(
        alg::Algorithm,
        boffs1::BlockOffsets,
        inds1,
        labels1,
        boffs2::BlockOffsets,
        inds2,
        labels2,
        indsR,
        labelsR
    )
    NR = length(labelsR)
    ValNR = ValLength(labelsR)
    labels1_to_labels2, labels1_to_labelsR, labels2_to_labelsR = contract_labels(
        labels1, labels2, labelsR
    )
    contraction_plan = contract_blocks(
        alg, boffs1, boffs2, labels1_to_labels2, labels1_to_labelsR, labels2_to_labelsR,
        ValNR
    )
    blockoffsetsR = BlockOffsets{NR}()
    nnzR = 0
    for (_, _, blockR) in contraction_plan
        if !isassigned(blockoffsetsR, blockR)
            insert!(blockoffsetsR, blockR, nnzR)
            nnzR += blockdim(indsR, blockR)
        end
    end
    return blockoffsetsR, contraction_plan
end

# A generic version making use of `Folds.jl` which
# can take various Executor backends.
# Used for sequential and threaded contract functions.
function contract_blocksparse_with_executor!(
        R::BlockSparseTensor,
        labelsR,
        tensor1::BlockSparseTensor,
        labelstensor1,
        tensor2::BlockSparseTensor,
        labelstensor2,
        contraction_plan,
        executor,
        α::Number = one(eltype(R)),
        β::Number = zero(eltype(R))
    )
    # Group the contraction plan by the output block,
    # since the sets of contractions into the same block
    # must be performed sequentially to reduce over those
    # sets of contractions properly (and avoid race conditions).
    # Same as:
    # ```julia
    # grouped_contraction_plan = group(last, contraction_plan)
    # ```
    # but more efficient since we know the groups/keys already,
    # since they are the nonzero blocks of the output tensor `R`.
    grouped_contraction_plan = map(_ -> empty(contraction_plan), eachnzblock(R))
    for block_contraction in contraction_plan
        push!(grouped_contraction_plan[last(block_contraction)], block_contraction)
    end
    _contract_blocksparse_grouped!(
        R, labelsR, tensor1, labelstensor1, tensor2, labelstensor2,
        grouped_contraction_plan, executor, α, β
    )
    return R
end
# Function barrier to improve type stability,
# since `Folds`/`FLoops` is not type stable:
# https://discourse.julialang.org/t/type-instability-in-floop-reduction/68598
function _contract_blocksparse_grouped!(
        R::BlockSparseTensor,
        labelsR,
        tensor1::BlockSparseTensor,
        labelstensor1,
        tensor2::BlockSparseTensor,
        labelstensor2,
        grouped_contraction_plan,
        executor,
        α::Number = one(eltype(R)),
        β::Number = zero(eltype(R))
    )
    Folds.foreach(grouped_contraction_plan.values, executor) do contraction_plan_group
        # On the first write to each output block, scale the existing R
        # contribution by the outer β (so `R = α * (T1 * T2) + β * R`);
        # subsequent contributions to the same output block accumulate
        # (β = 1).
        β_block = β
        for block_contraction in contraction_plan_group
            blocktensor1, blocktensor2, blockR = block_contraction

            # <fermions>: per-block fermion sign, multiplied into the
            # outer `α` so the inner per-block contract scales the
            # contribution by `α * fermion_sign`.
            α_block =
                α * compute_alpha(
                eltype(R),
                labelsR,
                blockR,
                inds(R),
                labelstensor1,
                blocktensor1,
                inds(tensor1),
                labelstensor2,
                blocktensor2,
                inds(tensor2)
            )

            contract!(
                R[blockR],
                labelsR,
                tensor1[blocktensor1],
                labelstensor1,
                tensor2[blocktensor2],
                labelstensor2,
                α_block,
                β_block
            )

            # After the first contribution lands in this output block,
            # subsequent contributions accumulate.
            β_block = one(eltype(R))
        end
    end
    return nothing
end
