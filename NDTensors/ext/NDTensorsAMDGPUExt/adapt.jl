using NDTensors.AMDGPUExtensions: AMDGPUExtensions, ROCArrayAdaptor
using NDTensors.GPUArraysCoreExtensions: storagemode
function AMDGPUExtensions.roc(xs)
  return fmap(x -> adapt(ROCArrayAdaptor{AMDGPU.Runtime.Mem.HIPBuffer}(), x), xs)
end

function Adapt.adapt_storage(adaptor::ROCArrayAdaptor, xs::AbstractArray)
  ElT = eltype(xs)
  BufT = stro(adaptor)
  N = ndims(xs)
  return isbits(xs) ? xs : adapt(ROCArray{ElT,N,BufT}, xs)
end

function NDTensors.adapt_storagetype(
  adaptor::ROCArrayAdaptor, xs::Type{EmptyStorage{ElT,StoreT}}
) where {ElT,StoreT}
  BufT = storagemode(adaptor)
  return NDTensors.emptytype(NDTensors.adapt_storagetype(ROCVector{ElT,BufT}, StoreT))
end
