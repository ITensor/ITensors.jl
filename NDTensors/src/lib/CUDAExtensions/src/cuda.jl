using NDTensors.TypeParameterAccessors: TypeParameterAccessors, Position
using NDTensors.GPUArraysCoreExtensions: storagemode
# Implemented in NDTensorsCUDAExt
function cu end

## Here we need an CuArrayAdaptor because the CuArrayAdaptor provided by CUDA
## converts 64 bit numbers to 32 bit.  We cannot write `adapt(CuVector, x)` because this
## Will not allow us to properly utilize the buffer preference without changing the value of
## default_buffertype. Also `adapt(CuVector{<:Any, <:Any, Buffertype})` fails to work properly
struct CuArrayAdaptor{B} end

function TypeParameterAccessors.position(::Type{<:CuArrayAdaptor}, ::typeof(storagemode))
  return Position(1)
end
