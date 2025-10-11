using Adapt: Adapt, adapt
using Functors: fmap
using Metal: MtlArray, MtlVector, DefaultStorageMode
using NDTensors: NDTensors, EmptyStorage, adapt_storagetype, emptytype
using NDTensors.Expose: Exposed
using NDTensors.MetalExtensions: MetalExtensions, MtlArrayAdaptor
using NDTensors.GPUArraysCoreExtensions: GPUArraysCoreExtensions
using NDTensors.Vendored.TypeParameterAccessors: set_type_parameters, type_parameters

GPUArraysCoreExtensions.cpu(e::Exposed{<:MtlArray}) = adapt(Array, e)

function MetalExtensions.mtl(xs; storagemode = DefaultStorageMode)
    return fmap(x -> adapt(MtlArrayAdaptor{storagemode}(), x), xs)
end

function Adapt.adapt_storage(adaptor::MtlArrayAdaptor, xs::AbstractArray)
    new_parameters = (type_parameters(xs, (eltype, ndims))..., storagemode(adaptor))
    mtltype = set_type_parameters(MtlArray, (eltype, ndims, storagemode), new_parameters)
    return isbits(xs) ? xs : adapt(mtltype, xs)
end

function NDTensors.adapt_storagetype(
        adaptor::MtlArrayAdaptor, ::Type{EmptyStorage{ElT, StoreT}}
    ) where {ElT, StoreT}
    mtltype = set_type_parameters(
        MtlVector, (eltype, storagemode), (ElT, storagemode(adaptor))
    )
    return emptytype(adapt_storagetype(mtltype, StoreT))
end
