using Adapt: adapt
using GPUArraysCore: AbstractGPUArray
using NDTensors: NDTensors, Dense, DenseTensor, Diag, DiagTensor, NativeContract, Tensor,
    contract!, dense, inds
using TypeParameterAccessors: set_ndims, unwrap_array_type

# GPU dispatch for `Diag × Dense` (and `Dense × Diag`) shapes routed
# through `NativeContract`. Without these overrides, the inner CPU body
# of `contract!(::NativeContract, C::DenseTensor, ..., A::DiagTensor,
# ..., B::DenseTensor, ...)` does scalar iteration over the diagonal,
# which on a GPU array errors with "scalar indexing is disallowed".
# These methods convert the diagonal tensor to a GPU-backed dense
# tensor and recurse to the dense × dense `NativeContract` path.

# `DiagTensor × DenseTensor`, both backed by GPU arrays.
function NDTensors.contract!(
        ::NativeContract,
        output_tensor::DenseTensor{<:Any, <:Any, <:Any, <:Dense{<:Any, <:AbstractGPUArray}},
        labelsoutput_tensor,
        tensor1::DiagTensor{<:Any, <:Any, <:Any, <:Diag{<:Any, <:AbstractGPUArray}},
        labelstensor1,
        tensor2::DenseTensor{<:Any, <:Any, <:Any, <:Dense{<:Any, <:AbstractGPUArray}},
        labelstensor2,
        α::Number = one(Bool),
        β::Number = zero(Bool)
    )
    tensor1 = dense(tensor1)
    contract!(
        NativeContract(),
        output_tensor, labelsoutput_tensor, tensor1, labelstensor1, tensor2, labelstensor2,
        α, β
    )
    return output_tensor
end

# `DenseTensor × DiagTensor`, both GPU. Swap to put the diagonal first.
function NDTensors.contract!(
        ::NativeContract,
        output_tensor::DenseTensor{<:Any, <:Any, <:Any, <:Dense{<:Any, <:AbstractGPUArray}},
        labelsoutput_tensor,
        tensor1::DenseTensor{<:Any, <:Any, <:Any, <:Dense{<:Any, <:AbstractGPUArray}},
        labelstensor1,
        tensor2::DiagTensor{<:Any, <:Any, <:Any, <:Diag{<:Any, <:AbstractGPUArray}},
        labelstensor2,
        α::Number = one(Bool),
        β::Number = zero(Bool)
    )
    contract!(
        NativeContract(),
        output_tensor, labelsoutput_tensor, tensor2, labelstensor2, tensor1, labelstensor1,
        α, β
    )
    return output_tensor
end

# `UniformDiag × DenseTensor`: tensor1 is a `DiagTensor` with scalar
# (`<:Number`) storage, tensor2 is GPU-backed dense. Convert the
# uniform diagonal to a GPU dense tensor of matching array type and
# recurse.
function NDTensors.contract!(
        ::NativeContract,
        output_tensor::DenseTensor{<:Any, <:Any, <:Any, <:Dense{<:Any, <:AbstractGPUArray}},
        labelsoutput_tensor,
        tensor1::DiagTensor{<:Any, <:Any, <:Any, <:Diag{<:Any, <:Number}},
        labelstensor1,
        tensor2::DenseTensor{<:Any, <:Any, <:Any, <:Dense{<:Any, <:AbstractGPUArray}},
        labelstensor2,
        α::Number = one(Bool),
        β::Number = zero(Bool)
    )
    # TODO: this allocates on CPU first then moves to GPU; could be
    # optimized by allocating directly on the device.
    tensor1 = adapt(set_ndims(unwrap_array_type(tensor2), 1), dense(tensor1))
    contract!(
        NativeContract(),
        output_tensor, labelsoutput_tensor, tensor1, labelstensor1, tensor2, labelstensor2,
        α, β
    )
    return output_tensor
end

# `DenseTensor × UniformDiag`: swap to put the uniform diagonal first.
function NDTensors.contract!(
        ::NativeContract,
        output_tensor::DenseTensor{<:Any, <:Any, <:Any, <:Dense{<:Any, <:AbstractGPUArray}},
        labelsoutput_tensor,
        tensor1::DenseTensor{<:Any, <:Any, <:Any, <:Dense{<:Any, <:AbstractGPUArray}},
        labelstensor1,
        tensor2::DiagTensor{<:Any, <:Any, <:Any, <:Diag{<:Any, <:Number}},
        labelstensor2,
        α::Number = one(Bool),
        β::Number = zero(Bool)
    )
    contract!(
        NativeContract(),
        output_tensor, labelsoutput_tensor, tensor2, labelstensor2, tensor1, labelstensor1,
        α, β
    )
    return output_tensor
end
