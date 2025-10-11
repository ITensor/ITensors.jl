using ..Vendored.TypeParameterAccessors: TypeParameterAccessors, Position
using ..GPUArraysCoreExtensions: storagemode
# Implemented in NDTensorsMetalExt
function mtl end

## Here we need an MtlArrayAdaptor because the MtlArrayAdaptor provided by Metal
## converts 64 bit numbers to 32 bit.  We cannot write `adapt(MtlArray, x)` because this
## Will not allow us to properly utilize the buffer preference without changing the value of
## default_buffertype. Also `adapt(MtlArray{<:Any, <:Any, Buffertype})` fails to work properly

struct MtlArrayAdaptor{B} end

function TypeParameterAccessors.position(::Type{<:MtlArrayAdaptor}, ::typeof(storagemode))
    return Position(1)
end
