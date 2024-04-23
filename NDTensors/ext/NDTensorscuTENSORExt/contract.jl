using NDTensors: NDTensors, Tensor
using NDTensors.Expose: unexpose
using CUDA: CuArray
using cuTENSOR: cuTENSOR
function NDTensors.contract(Etensor1::Exposed{<:CuArray}, labelstensor1, Etensor2::Exposed{<:CuArray}, labelstensor2, labelsoutput_tensor)
  tensor1 = unexpose(Etensor1)
  tensor2 = unexpose(Etensor2)
  ## reshape the flat data from ITensors into the correct dimensions
  dA = reshape(data(tensor1), dims(tensor1))
  dB = reshape(data(tensor2), dims(tensor2))
  ## convert the ITensors into CuTensors
  cutensorA = CuTensor(dA, [i for i in labelstensor1])
  cutensorB = CuTensor(dB, [i for i in labelstensor2])

  ## contract the CuTensors
  cutensorC = cutensorA * cutensorB

  ## TODO this is a first draft to this idea to see if the 
  ## conversion works 
  output_tensor = NDTensors.contraction_output(
    tensor1, labelstensor1, tensor2, labelstensor2, labelsoutput_tensor
  )

  ## Replace the data in the output_tensor with the correct data from the cutensor contraction
  ## it is necessary to flatten the data
  output_tensor = NDTensors.setstorage(output_tensor, NDTensors.setdata(NDTensors.storage(output_tensor), reshape(cutensorC.data, dim(output_tensor))))

  ## return output_tensor
  return output_tensor
end