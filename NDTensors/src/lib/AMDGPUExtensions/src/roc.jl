using ..Vendored.TypeParameterAccessors: TypeParameterAccessors, Position
using ..GPUArraysCoreExtensions: storagemode
# Implemented in NDTensorsAMDGPUExt
function roc end

## Here we need an ROCArrayAdaptor to prevent conversion of 64 bit numbers to 32 bit.
## We cannot write `adapt(CuVector, x)` because this
## will not allow us to properly utilize the buffer preference without changing the value of
## default_buffertype. Also `adapt(CuVector{<:Any, <:Any, Buffertype})` fails to work properly
struct ROCArrayAdaptor{B} end

function TypeParameterAccessors.position(::Type{<:ROCArrayAdaptor}, ::typeof(storagemode))
    return Position(1)
end
