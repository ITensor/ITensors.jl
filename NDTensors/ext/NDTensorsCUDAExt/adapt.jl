## Here we need an NDTensorCuArrayAdaptor because the CuArrayAdaptor provided by CUDA
## converts 64 bit numbers to 32 bit.  We cannot write `adapt(CuVector, x)` because this
## Will not allow us to properly utilize the buffer preference without changing the value of
## default_buffertype. Also `adapt(CuVector{<:Any, <:Any, Buffertype})` fails to work properly
struct NDTensorCuArrayAdaptor{B} end

"""
cu(xs; Buffer::String="Device")

NDTensors version of `cu` function which preserves the number percision in the input.
The array will use the buffer type Buffer which has option of CUDA.Mem.DeviceBuffer
CUDA.Mem.UnifiedBuffer and CUDA.Mem.HostBuffer
"""
function cu(xs; buffertype=Mem.DeviceBuffer)
  return fmap(x -> adapt(NDTensorCuArrayAdaptor{buffertype}(), x), xs)
end

buffertype(buffer::NDTensorCuArrayAdaptor) = parameter(buffer)

function Adapt.adapt_storage(adaptor::NDTensorCuArrayAdaptor, xs::AbstractArray)
  ElT = eltype(xs)
  BufT = buffertype(adaptor)
  N = ndims(xs)
  return isbits(xs) ? xs : adapt(CuArray{ElT,N,BufT}, xs)
end

function NDTensors.adapt_storagetype(
  adaptor::NDTensorCuArrayAdaptor, xs::Type{EmptyStorage{ElT,StoreT}}
) where {ElT,StoreT}
  BufT = buffertype(adaptor)
  return NDTensors.emptytype(NDTensors.adapt_storagetype(CuVector{ElT,BufT}, StoreT))
end
