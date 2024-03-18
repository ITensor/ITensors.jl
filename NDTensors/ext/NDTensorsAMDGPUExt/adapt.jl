using NDTensors.AMDGPUExtensions: AMDGPUExtensions, RocArrayAdaptor
function AMDGPUExtensions.roc(xs)
  return fmap(x -> adapt(ROCArrayAdaptor{AMDGPU.Runtime.Mem.HIPBuffer}(), x), xs)
end

buffertype(::ROCArrayAdaptor{B}) where {B} = B

function Adapt.adapt_storage(adaptor::ROCArrayAdaptor, xs::AbstractArray)
  ElT = eltype(xs)
  BufT = buffertype(adaptor)
  N = ndims(xs)
  return isbits(xs) ? xs : adapt(ROCArray{ElT,N,BufT}, xs)
end

function NDTensors.adapt_storagetype(
  adaptor::ROCArrayAdaptor, xs::Type{EmptyStorage{ElT,StoreT}}
) where {ElT,StoreT}
  BufT = buffertype(adaptor)
  return NDTensors.emptytype(NDTensors.adapt_storagetype(ROCVector{ElT,BufT}, StoreT))
end
