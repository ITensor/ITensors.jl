using .Expose: expose
function contract_blocks!(
  alg::Algorithm"threaded_threads",
  contraction_plans,
  boffs1,
  boffs2,
  labels1_to_labels2,
  labels1_to_labelsR,
  labels2_to_labelsR,
  ValNR,
)
  blocks1 = keys(boffs1)
  blocks2 = keys(boffs2)
  if length(blocks1) > length(blocks2)
    @sync for blocks1_partition in
              Iterators.partition(blocks1, max(1, length(blocks1) ÷ nthreads()))
      @spawn for block1 in blocks1_partition
        for block2 in blocks2
          maybe_contract_blocks!(
            contraction_plans[threadid()],
            block1,
            block2,
            labels1_to_labels2,
            labels1_to_labelsR,
            labels2_to_labelsR,
            ValNR,
          )
        end
      end
    end
  else
    @sync for blocks2_partition in
              Iterators.partition(blocks2, max(1, length(blocks2) ÷ nthreads()))
      @spawn for block2 in blocks2_partition
        for block1 in blocks1
          maybe_contract_blocks!(
            contraction_plans[threadid()],
            block1,
            block2,
            labels1_to_labels2,
            labels1_to_labelsR,
            labels2_to_labelsR,
            ValNR,
          )
        end
      end
    end
  end
  return nothing
end
