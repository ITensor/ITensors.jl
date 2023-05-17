function contract(
  tensor1::BlockSparseTensor,
  labelstensor1,
  tensor2::BlockSparseTensor,
  labelstensor2,
  labelsR=contract_labels(labelstensor1, labelstensor2),
)
  R, contraction_plan = contraction_output(
    tensor1, labelstensor1, tensor2, labelstensor2, labelsR
  )
  R = contract!(
    R, labelsR, tensor1, labelstensor1, tensor2, labelstensor2, contraction_plan
  )
  return R
end

# Determine the contraction output and block contractions
function contraction_output(
  tensor1::BlockSparseTensor,
  labelstensor1,
  tensor2::BlockSparseTensor,
  labelstensor2,
  labelsR,
)
  indsR = contract_inds(inds(tensor1), labelstensor1, inds(tensor2), labelstensor2, labelsR)
  TensorR = contraction_output_type(typeof(tensor1), typeof(tensor2), indsR)
  blockoffsetsR, contraction_plan = contract_blockoffsets(
    blockoffsets(tensor1),
    inds(tensor1),
    labelstensor1,
    blockoffsets(tensor2),
    inds(tensor2),
    labelstensor2,
    indsR,
    labelsR,
  )
  R = similar(TensorR, blockoffsetsR, indsR)
  return R, contraction_plan
end

function contract_blockoffsets(
  boffs1::BlockOffsets, inds1, labels1, boffs2::BlockOffsets, inds2, labels2, indsR, labelsR
)
  alg = Algorithm"sequential"()
  if using_threaded_blocksparse() && nthreads() > 1
    alg = Algorithm"threaded_threads"()
    # This code is a bit cleaner but slower:
    # alg = Algorithm"threaded_folds"()
  end
  return contract_blockoffsets(
    alg, boffs1, inds1, labels1, boffs2, inds2, labels2, indsR, labelsR
  )
end

function contract!(
  R::BlockSparseTensor,
  labelsR,
  tensor1::BlockSparseTensor,
  labelstensor1,
  tensor2::BlockSparseTensor,
  labelstensor2,
  contraction_plan,
)
  if isempty(contraction_plan)
    return R
  end
  alg = Algorithm"sequential"()
  if using_threaded_blocksparse() && nthreads() > 1
    alg = Algorithm"threaded_folds"()
  end
  return contract!(
    alg, R, labelsR, tensor1, labelstensor1, tensor2, labelstensor2, contraction_plan
  )
end
