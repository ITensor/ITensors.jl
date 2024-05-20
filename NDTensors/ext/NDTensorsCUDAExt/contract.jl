using NDTensors: NDTensors, DenseTensor, DiagTensor, contract!, dense, inds, Tensor
using NDTensors.Expose: Exposed, expose, unexpose
using NDTensors.TypeParameterAccessors: parenttype
using CUDA: CuArray

## In this function we convert the DiagTensor to a dense tensor and
## Feed it back into contract
function NDTensors.contract!(
    output_tensor::Exposed{<:CuArray, <:DenseTensor},
    labelsoutput_tensor,
    tensor1::Exposed{<:Any, <:DiagTensor},
    labelstensor1,
    tensor2::Exposed{<:CuArray, <:DenseTensor},
    labelstensor2,
    α::Number=one(Bool),
    β::Number=zero(Bool),
  )
    tensor1 = unexpose(tensor1)
    ## convert tensor1 to a dense
    tensor1 = adapt(parenttype(typeof(tensor2)), dense(tensor1))
    return contract!(
      output_tensor,
      labelsoutput_tensor,
      expose(tensor1),
      labelstensor1,
      tensor2,
      labelstensor2,
      α,
      β,
    )
  end

  function NDTensors.contract!(
    output_tensor::Exposed{<:CuArray, <:DenseTensor},
    labelsoutput_tensor,
    tensor1::Exposed{<:CuArray, <:DenseTensor},
    labelstensor1,
    tensor2::Exposed{<:Any, <:DiagTensor},
    labelstensor2,
    α::Number=one(Bool),
    β::Number=zero(Bool),
  )
    contract!(output_tensor, labelsoutput_tensor, tensor2, labelstensor2, tensor1, labelstensor1, α, β)
  end