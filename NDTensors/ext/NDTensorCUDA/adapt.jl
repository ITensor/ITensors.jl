#
# Used to adapt `EmptyStorage` types
#

## Here we need an NDTensorCuArrayAdaptor because the CuArrayAdaptor provided by CUDA
## converts 64 bit numbers to 32 bit.  We cannot write `adapt(CuVector, x)` because this
## Will not allow us to properly utilize the buffer preference without changing the value of
## default_buffertype. Also `adapt(CuVector{<:Any, <:Any, Buffertype})` fails to work properly
struct NDTensorCuArrayAdaptor{B} end
## TODO make this work for unified. This works but overwrites CUDA's adapt_storage. This fails for emptystorage...
@inline function NDTensors.cu(xs; unified::Bool=false)
  return fmap(
    x -> adapt(NDTensorCuArrayAdaptor{unified ? Mem.UnifiedBuffer : Mem.DeviceBuffer}(), x),
    xs,
  )
end

function Adapt.adapt_storage(
  ::NDTensorCuArrayAdaptor{B}, xs::AbstractArray{T,N}
) where {T,N,B}
  return isbits(xs) ? xs : CuArray{T,1,B}(xs)
end

function NDTensors.adapt_storagetype(
  ::NDTensorCuArrayAdaptor{B}, xs::Type{EmptyStorage{ElT,StoreT}}
) where {ElT,StoreT,B}
  return NDTensors.emptytype(NDTensors.adapt_storagetype(CuVector{ElT,B}, StoreT))
end
