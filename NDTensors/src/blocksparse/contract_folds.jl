# Function barrier to improve type stability,
# since `Folds`/`FLoops` is not type stable:
# https://discourse.julialang.org/t/type-instability-in-floop-reduction/68598
function contract_blocks!(
  alg::Algorithm"threaded_folds",
  contraction_plans,
  boffs1,
  boffs2,
  labels1_to_labels2,
  labels1_to_labelsR,
  labels2_to_labelsR,
  ValNR,
)
  if nnzblocks(boffs1) > nnzblocks(boffs2)
    Folds.foreach(eachnzblock(boffs1).values, ThreadedEx()) do block1
      for block2 in eachnzblock(boffs2)
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
  else
    Folds.foreach(eachnzblock(boffs2).values, ThreadedEx()) do block2
      for block1 in eachnzblock(boffs1)
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
  return nothing
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
