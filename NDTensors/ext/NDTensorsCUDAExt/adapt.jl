using Adapt: Adapt
using CUDA: CUDA, CuArray, CuVector
using Functors: fmap
using NDTensors: NDTensors, EmptyStorage, adapt_storagetype, emptytype
using NDTensors.CUDAExtensions: CUDAExtensions, CuArrayAdaptor
using NDTensors.GPUArraysCoreExtensions: storagemode
using NDTensors.Vendored.TypeParameterAccessors:
    default_type_parameters, set_type_parameters, type_parameters

function CUDAExtensions.cu(xs; storagemode = default_type_parameters(CuArray, storagemode))
    return fmap(x -> adapt(CuArrayAdaptor{storagemode}(), x), xs)
end

## Could do this generically
function Adapt.adapt_storage(adaptor::CuArrayAdaptor, xs::AbstractArray)
    params = (type_parameters(xs, (eltype, ndims))..., storagemode(adaptor))
    cutype = set_type_parameters(CuArray, (eltype, ndims, storagemode), params)
    return isbits(xs) ? xs : adapt(cutype, xs)
end

function NDTensors.adapt_storagetype(
        adaptor::CuArrayAdaptor, ::Type{EmptyStorage{ElT, StoreT}}
    ) where {ElT, StoreT}
    cutype = set_type_parameters(CuVector, (eltype, storagemode), (ElT, storagemode(adaptor)))
    return emptytype(adapt_storagetype(cutype, StoreT))
end
