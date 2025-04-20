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
