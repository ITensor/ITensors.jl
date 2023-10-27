function contraction_output(
  tensor1::MatrixOrArrayStorageTensor, tensor2::CombinerTensor, indsR
)
  tensortypeR = contraction_output_type(typeof(tensor1), typeof(tensor2), indsR)
  return NDTensors.similar(tensortypeR, indsR)
end
