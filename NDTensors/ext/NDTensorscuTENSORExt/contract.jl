using NDTensors: NDTensors, Tensor, array, inds
using NDTensors.Expose: Exposed, unexpose
using cuTENSOR: cuTENSOR, CuArray, CuTensor
function NDTensors.contract(
  Etensor1::Exposed{<:CuArray},
  labelstensor1,
  Etensor2::Exposed{<:CuArray},
  labelstensor2,
  labelsoutput_tensor,
)
  tensor1 = unexpose(Etensor1)
  tensor2 = unexpose(Etensor2)
  ## convert the ITensors into CuTensors
  ## for 
  cutensorA = CuTensor(convert(CuArray, array(tensor1)), collect(labelstensor1))
  cutensorB = CuTensor(convert(CuArray, array(tensor1)), collect(labelstensor2))

  ## contract the CuTensors
  cutensorC = cutensorA * cutensorB

  ## TODO this is a first draft to this idea to see if the 
  ## conversion works 
  indsR = contract_inds(inds(tensor1), labelstensor1, inds(tensor2), labelstensor2, labelsoutput_tensor)
  TensorR = contraction_output_type(typeof(tensor1), typeof(tensor2), labelsoutput_tensor)

  ## Replace the data in the output_tensor with the correct data from the cutensor contraction
  ## it is necessary to flatten the data
  output_tensor = TensorR(reshape(cutensorC.data, dim(output_tensor)), indsR)

  ## return output_tensor
  return output_tensor
end
