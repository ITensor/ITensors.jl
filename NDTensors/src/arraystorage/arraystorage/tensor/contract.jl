# TODO: Just call `contraction_output(storage(tensor1), storage(tensor2), indsR)`
function contraction_output(
  tensor1::MatrixOrArrayStorageTensor, tensor2::MatrixOrArrayStorageTensor, indsR
)
  tensortypeR = contraction_output_type(typeof(tensor1), typeof(tensor2), indsR)
  return NDTensors.similar(tensortypeR, indsR)
end

function contract!(
  tensorR::MatrixOrArrayStorageTensor,
  labelsR,
  tensor1::MatrixOrArrayStorageTensor,
  labels1,
  tensor2::MatrixOrArrayStorageTensor,
  labels2,
)
  contract!(storage(tensorR), labelsR, storage(tensor1), labels1, storage(tensor2), labels2)
  return tensorR
end
