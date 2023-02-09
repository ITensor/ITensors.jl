function contract_blockoffsets(
  alg::Algorithm"threaded_folds",
  boffs1::BlockOffsets,
  inds1,
  labels1,
  boffs2::BlockOffsets,
  inds2,
  labels2,
  indsR,
  labelsR,
)
  NR = length(labelsR)
  ValNR = ValLength(labelsR)
  labels1_to_labels2, labels1_to_labelsR, labels2_to_labelsR = contract_labels(
    labels1, labels2, labelsR
  )

  T = Tuple{blocktype(boffs1),blocktype(boffs2),Block{NR}}

  # Thread-local collections of block contractions.
  # Could use:
  # ```julia
  # FLoops.@reduce(contraction_plans = append!(T[], [(block1, block2, blockR)]))
  # ```
  # as a simpler alternative but it is slower.
  contraction_plans = Vector{T}[Vector{T}() for _ in 1:nthreads()]

  # # Reserve some capacity.
  # # In theory the maximum is `length(boffs1) * length(boffs2)`
  # # but in practice that is too much.
  # # This didn't seem to help the performance.
  # for contraction_plan in contraction_plans
  #   sizehint!(contraction_plan, max(length(boffs1), length(boffs2)))
  # End

  _contract_blocks!(
    alg,
    contraction_plans,
    boffs1,
    boffs2,
    labels1_to_labels2,
    labels1_to_labelsR,
    labels2_to_labelsR,
    ValNR,
  )

  # Collect the results across threads
  contraction_plan = reduce(vcat, contraction_plans)

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

# Function barrier to improve type stability,
# since `Folds`/`FLoops` is not type stable:
# https://discourse.julialang.org/t/type-instability-in-floop-reduction/68598
function _contract_blocks!(
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
        _maybe_contract_blocks!(
          alg,
          contraction_plans,
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
        _maybe_contract_blocks!(
          alg,
          contraction_plans,
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

function _maybe_contract_blocks!(
  ::Algorithm"threaded_folds",
  contraction_plans,
  block1,
  block2,
  labels1_to_labels2,
  labels1_to_labelsR,
  labels2_to_labelsR,
  ValNR,
)
  if are_blocks_contracted(block1, block2, labels1_to_labels2)
    blockR = contract_blocks(block1, labels1_to_labelsR, block2, labels2_to_labelsR, ValNR)
    block_contraction = (block1, block2, blockR)
    push!(contraction_plans[threadid()], block_contraction)
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
