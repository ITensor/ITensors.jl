using Adapt: Adapt
using CUDA: CUDA, CuArray
using Functors: fmap
using NDTensors: NDTensors, EmptyStorage
using NDTensors.CUDAExtensions: CUDAExtensions, CuArrayAdaptor
using NDTensors.GPUArraysCoreExtensions: storagemode
using NDTensors.TypeParameterAccessors: TypeParameterAccessors, default_type_parameter, set_type_parameters, type_parameters

function CUDAExtensions.cu(xs; storagemode=default_type_parameter(CuArray, storagemode))
  return fmap(
    x -> adapt(CuArrayAdaptor{storagemode}, x), xs
  )
end

## Could do this generically
function Adapt.adapt_storage(adaptor::CuArrayAdaptor, xs::AbstractArray)
  params = (type_parameters(xs, (eltype, ndims))..., storagemode(adaptor))
  cutype = set_type_parameters(CuArray, (eltype, ndims, storagemode), params)
  return isbits(xs) ? xs : adapt(cutype, xs)
end

function NDTensors.adapt_storagetype(
  adaptor::CuArrayAdaptor, xs::Type{EmptyStorage{ElT,StoreT}}
) where {ElT,StoreT}
  BufT = storagemode(adaptor)
  return NDTensors.emptytype(NDTensors.adapt_storagetype(CuVector{ElT,BufT}, StoreT))
end
