function contract_blockoffsets(
        ::Algorithm"sequential",
        boffs1::BlockOffsets,
        inds1,
        labels1,
        boffs2::BlockOffsets,
        inds2,
        labels2,
        indsR,
        labelsR
    )
    N1 = length(blocktype(boffs1))
    N2 = length(blocktype(boffs2))
    NR = length(labelsR)
    ValNR = ValLength(labelsR)
    labels1_to_labels2, labels1_to_labelsR, labels2_to_labelsR = contract_labels(
        labels1, labels2, labelsR
    )
    blockoffsetsR = BlockOffsets{NR}()
    nnzR = 0
    contraction_plan = Tuple{Block{N1}, Block{N2}, Block{NR}}[]
    # Reserve some capacity
    # In theory the maximum is length(boffs1) * length(boffs2)
    # but in practice that is too much
    sizehint!(contraction_plan, max(length(boffs1), length(boffs2)))
    for block1 in keys(boffs1)
        for block2 in keys(boffs2)
            if are_blocks_contracted(block1, block2, labels1_to_labels2)
                blockR = contract_blocks(
                    block1, labels1_to_labelsR, block2, labels2_to_labelsR, ValNR
                )
                push!(contraction_plan, (block1, block2, blockR))
                if !isassigned(blockoffsetsR, blockR)
                    insert!(blockoffsetsR, blockR, nnzR)
                    nnzR += blockdim(indsR, blockR)
                end
            end
        end
    end
    return blockoffsetsR, contraction_plan
end

function contract_blocksparse_sequential!(
        R::BlockSparseTensor,
        labelsR,
        tensor1::BlockSparseTensor,
        labelstensor1,
        tensor2::BlockSparseTensor,
        labelstensor2,
        contraction_plan,
        α::Number = one(eltype(R)),
        β::Number = zero(eltype(R))
    )
    executor = SequentialEx()
    return contract_blocksparse_with_executor!(
        R, labelsR, tensor1, labelstensor1, tensor2, labelstensor2, contraction_plan,
        executor, α, β
    )
end
