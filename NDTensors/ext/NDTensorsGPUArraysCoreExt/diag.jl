using GPUArraysCore: AbstractGPUArray
using NDTensors: NDTensors, BlockSparseTensor, dense, diag
using NDTensors.Expose: Exposed, unexpose

## TODO to circumvent issues with blocksparse and scalar indexing
## convert blocksparse GPU tensors to dense tensors and call diag
## copying will probably have some impact on timing but this code
## currently isn't used in the main code, just in tests.
function NDTensors.diag(ETensor::Exposed{<:AbstractGPUArray,<:BlockSparseTensor})
  return diag(dense(unexpose(ETensor)))
end
