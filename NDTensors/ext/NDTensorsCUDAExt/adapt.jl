using NDTensors.TypeParameterAccessors: TypeParameterAccessors
using NDTensors.GPUArraysCoreExtensions: storagemode
using NDTensors.CUDAExtensions: CUDAExtensions, CuArrayAdaptor

## TODO make this work for unified. This works but overwrites CUDA's adapt_storage. This fails for emptystorage...
function CUDAExtensions.cu(xs; unified::Bool=false)
  return fmap(
    x -> adapt(CuArrayAdaptor{unified ? Mem.UnifiedBuffer : Mem.DeviceBuffer}(), x), xs
  )
end

function Adapt.adapt_storage(adaptor::CuArrayAdaptor, xs::AbstractArray)
  ElT = eltype(xs)
  BufT = storagemode(adaptor)
  N = ndims(xs)
  return isbits(xs) ? xs : adapt(CuArray{ElT,N,BufT}, xs)
end

function NDTensors.adapt_storagetype(
  adaptor::CuArrayAdaptor, xs::Type{EmptyStorage{ElT,StoreT}}
) where {ElT,StoreT}
  BufT = storagemode(adaptor)
  return NDTensors.emptytype(NDTensors.adapt_storagetype(CuVector{ElT,BufT}, StoreT))
end
