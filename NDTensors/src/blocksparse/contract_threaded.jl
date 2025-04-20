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
  return if length(blocks1) > length(blocks2)
    tasks = map(
      Iterators.partition(blocks1, max(1, length(blocks1) ÷ nthreads()))
    ) do blocks1_partition
      @spawn begin
        block_contractions = Tuple{Block{N1},Block{N2},Block{NR}}[]
        foreach(Iterators.product(blocks1_partition, blocks2)) do (block1, block2)
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
        return block_contractions
      end
    end
    mapreduce(fetch, vcat, tasks)
  else
    tasks = map(
      Iterators.partition(blocks2, max(1, length(blocks2) ÷ nthreads()))
    ) do blocks2_partition
      @spawn begin
        block_contractions = Tuple{Block{N1},Block{N2},Block{NR}}[]
        foreach(Iterators.product(blocks1, blocks2_partition)) do (block1, block2)
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
        return block_contractions
      end
    end
    mapreduce(fetch, vcat, tasks)
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
