using Adapt: adapt
using GPUArraysCore: AbstractGPUArray
using NDTensors: NDTensors, DenseTensor, DiagTensor, contract!, dense, inds, Tensor
using NDTensors.Expose: Exposed, expose, unexpose
using NDTensors.Vendored.TypeParameterAccessors: parenttype, set_ndims

function NDTensors.contract!(
        output_tensor::Exposed{<:AbstractGPUArray, <:DenseTensor},
        labelsoutput_tensor,
        tensor1::Exposed{<:AbstractGPUArray, <:DiagTensor},
        labelstensor1,
        tensor2::Exposed{<:AbstractGPUArray, <:DenseTensor},
        labelstensor2,
        α::Number = one(Bool),
        β::Number = zero(Bool);
        convert_to_dense::Bool = true,
    )
    # Convert tensor1 to dense.
    # TODO: Define `Exposed` overload for `dense`.
    tensor1 = expose(dense(unexpose(tensor1)))
    contract!(
        output_tensor, labelsoutput_tensor, tensor1, labelstensor1, tensor2, labelstensor2, α, β
    )
    return output_tensor
end

function NDTensors.contract!(
        output_tensor::Exposed{<:AbstractGPUArray, <:DenseTensor},
        labelsoutput_tensor,
        tensor1::Exposed{<:AbstractGPUArray, <:DenseTensor},
        labelstensor1,
        tensor2::Exposed{<:AbstractGPUArray, <:DiagTensor},
        labelstensor2,
        α::Number = one(Bool),
        β::Number = zero(Bool),
    )
    contract!(
        output_tensor, labelsoutput_tensor, tensor2, labelstensor2, tensor1, labelstensor1, α, β
    )
    return output_tensor
end

## In this function we convert the DiagTensor to a dense tensor and
## Feed it back into contract
function NDTensors.contract!(
        output_tensor::Exposed{<:AbstractGPUArray, <:DenseTensor},
        labelsoutput_tensor,
        tensor1::Exposed{<:Number, <:DiagTensor},
        labelstensor1,
        tensor2::Exposed{<:AbstractGPUArray, <:DenseTensor},
        labelstensor2,
        α::Number = one(Bool),
        β::Number = zero(Bool),
    )
    # Convert tensor1 to dense.
    # TODO: Define `Exposed` overload for `dense`.
    # TODO: This allocates on CPU first then moves over to GPU which could be optimized.
    tensor1 = expose(
        adapt(set_ndims(parenttype(typeof(tensor2)), 1), dense(unexpose(tensor1)))
    )
    contract!(
        output_tensor, labelsoutput_tensor, tensor1, labelstensor1, tensor2, labelstensor2, α, β
    )
    return output_tensor
end

function NDTensors.contract!(
        output_tensor::Exposed{<:AbstractGPUArray, <:DenseTensor},
        labelsoutput_tensor,
        tensor1::Exposed{<:AbstractGPUArray, <:DenseTensor},
        labelstensor1,
        tensor2::Exposed{<:Number, <:DiagTensor},
        labelstensor2,
        α::Number = one(Bool),
        β::Number = zero(Bool),
    )
    contract!(
        output_tensor, labelsoutput_tensor, tensor2, labelstensor2, tensor1, labelstensor1, α, β
    )
    return output_tensor
end
