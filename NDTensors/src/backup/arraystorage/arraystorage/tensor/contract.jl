# TODO: Just call `contraction_output(storage(tensor1), storage(tensor2), indsR)`
function contraction_output(
  tensor1::MatrixOrArrayStorageTensor, tensor2::MatrixOrArrayStorageTensor, indsR
)
  tensortypeR = contraction_output_type(typeof(tensor1), typeof(tensor2), indsR)
  return NDTensors.similar(tensortypeR, indsR)
end

# TODO: Define `default_α` and `default_β`.
function contract!(
  tensor_dest::MatrixOrArrayStorageTensor,
  labels_dest,
  tensor1::MatrixOrArrayStorageTensor,
  labels1,
  tensor2::MatrixOrArrayStorageTensor,
  labels2,
  α=one(eltype(tensor_dest)),
  β=zero(eltype(tensor_dest));
)
  contract!(
    storage(tensor_dest),
    labels_dest,
    storage(tensor1),
    labels1,
    storage(tensor2),
    labels2,
    α,
    β,
  )
  return tensor_dest
end
