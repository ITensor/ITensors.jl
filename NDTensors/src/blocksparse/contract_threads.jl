using .Expose: expose
function contract_blocks(
  alg::Algorithm"threaded_threads",
  boffs1,
  boffs2,
  labels1_to_labels2,
  labels1_to_labelsR,
  labels2_to_labelsR,
  ValNR,
)
  blocks1 = keys(boffs1)
  blocks2 = keys(boffs2)
  return if length(blocks1) > length(blocks2)
    tasks = map(
      Iterators.partition(blocks1, max(1, length(blocks1) ÷ nthreads()))
    ) do blocks1_partition
      @spawn begin
        block_contractions =
          map(Iterators.product(blocks1_partition, blocks2)) do (block1, block2)
            maybe_contract_blocks(
              block1,
              block2,
              labels1_to_labels2,
              labels1_to_labelsR,
              labels2_to_labelsR,
              ValNR,
            )
          end
        block_contractions = filter(!isnothing, block_contractions)
      end
    end
    mapreduce(fetch, vcat, tasks)
  else
    tasks = map(
      Iterators.partition(blocks2, max(1, length(blocks2) ÷ nthreads()))
    ) do blocks2_partition
      @spawn begin
        block_contractions =
          map(Iterators.product(blocks1, blocks2_partition)) do (block1, block2)
            maybe_contract_blocks(
              block1,
              block2,
              labels1_to_labels2,
              labels1_to_labelsR,
              labels2_to_labelsR,
              ValNR,
            )
          end
        block_contractions = filter(!isnothing, block_contractions)
      end
    end
    mapreduce(fetch, vcat, tasks)
  end
end
