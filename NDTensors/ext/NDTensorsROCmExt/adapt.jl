## Code adapted from NDTensors/ext/NDTensorsCUDAExt/adapt.jl
## Here we need an NDTensorROCArrayAdaptor because AMDGPU.jl only provides a Float32Adaptor

struct NDTensorROCArrayAdaptor{B} end

function roc(xs)
  return fmap(x -> adapt(NDTensorROCArrayAdaptor{AMDGPU.Runtime.Mem.HIPBuffer}(), x), xs)
end

buffertype(::NDTensorROCArrayAdaptor{B}) where {B} = B

function Adapt.adapt_storage(adaptor::NDTensorROCArrayAdaptor, xs::AbstractArray)
  ElT = eltype(xs)
  BufT = buffertype(adaptor)
  N = ndims(xs)
  return isbits(xs) ? xs : adapt(ROCArray{ElT,N,BufT}, xs)
end

function NDTensors.adapt_storagetype(
  adaptor::NDTensorROCArrayAdaptor, xs::Type{EmptyStorage{ElT,StoreT}}
) where {ElT,StoreT}
  BufT = buffertype(adaptor)
  return NDTensors.emptytype(NDTensors.adapt_storagetype(ROCVector{ElT,BufT}, StoreT))
end
