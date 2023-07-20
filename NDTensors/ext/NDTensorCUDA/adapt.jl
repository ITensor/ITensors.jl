## Here we need an NDTensorCuArrayAdaptor because the CuArrayAdaptor provided by CUDA
## converts 64 bit numbers to 32 bit.  We cannot write `adapt(CuVector, x)` because this
## Will not allow us to properly utilize the buffer preference without changing the value of
## default_buffertype. Also `adapt(CuVector{<:Any, <:Any, Buffertype})` fails to work properly
struct NDTensorCuArrayAdaptor{B} end
## TODO make this work for unified. This works but overwrites CUDA's adapt_storage. This fails for emptystorage...
function cu(xs; unified::Bool=false)
  return fmap(
    x -> adapt(NDTensorCuArrayAdaptor{unified ? Mem.UnifiedBuffer : Mem.DeviceBuffer}(), x),
    xs,
  )
end

buffertype(::NDTensorCuArrayAdaptor{B}) where {B} = B

function Adapt.adapt_storage(adaptor::NDTensorCuArrayAdaptor, xs::AbstractArray)
  ElT = eltype(xs)
  BufT = buffertype(adaptor)
  return isbits(xs) ? xs : CuArray{ElT,1,BufT}(xs)
end

function NDTensors.adapt_storagetype(
  adaptor::NDTensorCuArrayAdaptor, xs::Type{EmptyStorage{ElT,StoreT}}
) where {ElT,StoreT}
  BufT = buffertype(adaptor)
  return NDTensors.emptytype(NDTensors.adapt_storagetype(CuVector{ElT,BufT}, StoreT))
end
