using NDTensors: NDTensors, Tensor
using NDTensors.Expose: Exposed, unexpose
using CUDA: CuArray
using cuTENSOR: cuTENSOR
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
  cutensorA = CuTensor(array(tensor1), collect(labelstensor1))
  cutensorB = CuTensor(array(tensor2), collect(labelstensor2))

  ## contract the CuTensors
  cutensorC = cutensorA * cutensorB

  ## TODO this is a first draft to this idea to see if the 
  ## conversion works 
  output_tensor = NDTensors.contraction_output(
    tensor1, labelstensor1, tensor2, labelstensor2, labelsoutput_tensor
  )

  ## Replace the data in the output_tensor with the correct data from the cutensor contraction
  ## it is necessary to flatten the data
  output_tensor = NDTensors.setstorage(
    output_tensor,
    NDTensors.setdata(
      NDTensors.storage(output_tensor), reshape(cutensorC.data, dim(output_tensor))
    ),
  )

  ## return output_tensor
  return output_tensor
end
