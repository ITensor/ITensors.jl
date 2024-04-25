using NDTensors: NDTensors, Tensor, array, contraction_output_type, contract_inds, inds
using NDTensors.Expose: Exposed, unexpose
using cuTENSOR: cuTENSOR, CuArray, CuTensor
using Adapt: adapt

function NDTensors.contract(
  Etensor1::Exposed{<:CuArray},
  labelstensor1,
  Etensor2::Exposed{<:CuArray},
  labelstensor2,
  labelsoutput_tensor,
)
  tensor1 = unexpose(Etensor1)
  tensor2 = unexpose(Etensor2)
  elt = promote_type(eltype(tensor1), eltype(tensor2))
  tensor1 = adapt(elt, tensor1)
  tensor2 = adapt(elt, tensor2)
  ## convert the ITensors into CuTensors
  ## This can fail when array(tensor) returns a ReshapedArray(CuArray)
  cutensorA = CuTensor(array(tensor1), collect(labelstensor1))
  cutensorB = CuTensor(array(tensor2), collect(labelstensor2))

  ## contract the CuTensors
  cutensorC = cutensorA * cutensorB

  ## TODO this is a first draft to this idea to see if the 
  ## conversion works 
  indsR = contract_inds(
    inds(tensor1), labelstensor1, inds(tensor2), labelstensor2, labelsoutput_tensor
  )
  TensorR = contraction_output_type(typeof(tensor1), typeof(tensor2), indsR)

  ## Replace the data in the output_tensor with the correct data from the cutensor contraction
  ## it is necessary to flatten the data
  ## TODO this could possibly fail for BlockSparse so need to determine that
  output_tensor = TensorR(
    NDTensors.AllowAlias(),
    NDTensors.storagetype(TensorR)(reshape(cutensorC.data, length(cutensorC.data))),
    indsR,
  )

  ## return output_tensor
  return output_tensor
end
