using NDTensors:
  NDTensors,
  BlockSparseTensor,
  Tensor,
  array,
  contract!,
  contraction_output,
  contraction_output_type,
  contract_inds,
  dense,
  inds
using NDTensors.Expose: Exposed, expose, unexpose
using cuTENSOR: CuArray, CuTensor

function NDTensors.contract(
  Etensor1::Exposed{<:CuArray},
  labelstensor1,
  Etensor2::Exposed{<:CuArray},
  labelstensor2,
  labelsoutput_tensor,
)
  tensor1 = unexpose(Etensor1)
  tensor2 = unexpose(Etensor2)

  return cutensor_contract(
    tensor1, labelstensor1, tensor2, labelstensor2, labelsoutput_tensor
  )
end

## working to fix blocksparse implementation. 
function NDTensors.contract(
  Etensor1::Exposed{<:CuArray,<:BlockSparseTensor},
  labelstensor1,
  Etensor2::Exposed{<:CuArray,<:BlockSparseTensor},
  labelstensor2,
  labelsoutput_tensor,
)
  ## temporarily don't use cutensor here, just keep implemented to prevent error
  tensor1 = unexpose(tensor1)
  tensor2 = unexpose(tensor2)
  R, contraction_plan = contraction_output(
    tensor1, labelstensor1, tensor2, labelstensor2, labelsR
  )
  R = contract!(
    R, labelsR, tensor1, labelstensor1, tensor2, labelstensor2, contraction_plan
  )
  return R
  #   tensor1 = unexpose(Etensor1)
  #   tensor2 = unexpose(Etensor2)

  #   denseoutput_tensor = cutensor_contract(dense(tensor1), labelstensor1, dense(tensor2), labelstensor2, labelsoutput_tensor)

  #   ## transform the dense tensor back to blocksparse
  #   indsR = contract_inds(
  #     inds(tensor1), labelstensor1, inds(tensor2), labelstensor2, labelsoutput_tensor
  #   )
  #   TensorR = contraction_output_type(typeof(tensor1), typeof(tensor2), indsR)
  #   return to_sparse(TensorR, denseoutput_tensor, indsR)
end

## TODO this only works for dense tensors
function cutensor_contract(
  tensor1::Tensor, labelstensor1, tensor2::Tensor, labelstensor2, labelsoutput_tensor
)
  elt = promote_type(eltype(tensor1), eltype(tensor2))
  tensor1, tensor2 = promote_tensor_eltype(elt, tensor1, tensor2)
  ## convert the ITensors into CuTensors
  ## This can fail when array(tensor) returns a wrapped CuArray

  ## TODO write a cutensor converter function which only copies when 
  ## array(tensor) != CuArray i.e. use expose.
  cutensorA = CuTensor(copy(expose(array(tensor1))), collect(labelstensor1))
  cutensorB = CuTensor(copy(expose(array(tensor2))), collect(labelstensor2))

  ## contract the CuTensors
  cutensorC = cutensorA * cutensorB

  indsR = contract_inds(
    inds(tensor1), labelstensor1, inds(tensor2), labelstensor2, labelsoutput_tensor
  )
  TensorR = contraction_output_type(typeof(tensor1), typeof(tensor2), indsR)

  ## Replace the data in the output_tensor with the correct data from the cutensor contraction
  ## it is necessary to flatten the data
  output_tensor = TensorR(
    NDTensors.AllowAlias(),
    NDTensors.storagetype(TensorR)(reshape(cutensorC.data, length(cutensorC.data))),
    indsR,
  )

  ## return output_tensor
  return output_tensor
end
