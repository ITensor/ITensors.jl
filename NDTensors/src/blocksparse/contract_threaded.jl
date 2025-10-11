using .Expose: expose
function contract_blocks(
        alg::Algorithm"threaded_threads",
        boffs1,
        boffs2,
        labels1_to_labels2,
        labels1_to_labelsR,
        labels2_to_labelsR,
        ValNR::Val{NR},
    ) where {NR}
    N1 = length(blocktype(boffs1))
    N2 = length(blocktype(boffs2))
    blocks1 = keys(boffs1)
    blocks2 = keys(boffs2)
    T = Tuple{Block{N1}, Block{N2}, Block{NR}}
    return if length(blocks1) > length(blocks2)
        tasks = map(
            Iterators.partition(blocks1, max(1, length(blocks1) ÷ nthreads()))
        ) do blocks1_partition
            @spawn begin
                block_contractions = T[]
                for block1 in blocks1_partition
                    for block2 in blocks2
                        block_contraction = maybe_contract_blocks(
                            block1,
                            block2,
                            labels1_to_labels2,
                            labels1_to_labelsR,
                            labels2_to_labelsR,
                            ValNR,
                        )
                        if !isnothing(block_contraction)
                            push!(block_contractions, block_contraction)
                        end
                    end
                end
                return block_contractions
            end
        end
        all_block_contractions = T[]
        for task in tasks
            append!(all_block_contractions, fetch(task))
        end
        return all_block_contractions
    else
        tasks = map(
            Iterators.partition(blocks2, max(1, length(blocks2) ÷ nthreads()))
        ) do blocks2_partition
            @spawn begin
                block_contractions = T[]
                for block2 in blocks2_partition
                    for block1 in blocks1
                        block_contraction = maybe_contract_blocks(
                            block1,
                            block2,
                            labels1_to_labels2,
                            labels1_to_labelsR,
                            labels2_to_labelsR,
                            ValNR,
                        )
                        if !isnothing(block_contraction)
                            push!(block_contractions, block_contraction)
                        end
                    end
                end
                return block_contractions
            end
        end
        all_block_contractions = T[]
        for task in tasks
            append!(all_block_contractions, fetch(task))
        end
        return all_block_contractions
    end
end

function contract!(
        ::Algorithm"threaded_folds",
        R::BlockSparseTensor,
        labelsR,
        tensor1::BlockSparseTensor,
        labelstensor1,
        tensor2::BlockSparseTensor,
        labelstensor2,
        contraction_plan,
    )
    executor = ThreadedEx()
    return contract!(
        R, labelsR, tensor1, labelstensor1, tensor2, labelstensor2, contraction_plan, executor
    )
end
