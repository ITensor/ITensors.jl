using NDTensors.TypeParameterAccessors: TypeParameterAccessors
using NDTensors.GPUArraysCoreExtensions: storagemode
# Implemented in `ITensorGPU` and NDTensorsCUDAExt
function cu end

## Here we need an CuArrayAdaptor because the CuArrayAdaptor provided by CUDA
## converts 64 bit numbers to 32 bit.  We cannot write `adapt(CuVector, x)` because this
## Will not allow us to properly utilize the buffer preference without changing the value of
## default_buffertype. Also `adapt(CuVector{<:Any, <:Any, Buffertype})` fails to work properly
struct CuArrayAdaptor{B} end

## TODO remove TypeParameterAccessors when SetParameters is removed
function TypeParameterAccessors.position(::Type{<:CuArrayAdaptor}, ::typeof(storagemode))
  return TypeParameterAccessors.Position(1)
end
