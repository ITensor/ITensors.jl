using NDTensors: NDTensors, EmptyStorage, adapt_storagetype, emptytype
using NDTensors.AMDGPUExtensions: AMDGPUExtensions, ROCArrayAdaptor
using NDTensors.GPUArraysCoreExtensions: storagemode
using NDTensors.Vendored.TypeParameterAccessors:
    default_type_parameters, set_type_parameters, type_parameters
using Adapt: Adapt, adapt
using AMDGPU: AMDGPU, ROCArray, ROCVector
using Functors: fmap

function AMDGPUExtensions.roc(
        xs; storagemode = default_type_parameters(ROCArray, storagemode)
    )
    return fmap(x -> adapt(ROCArrayAdaptor{storagemode}(), x), xs)
end

function Adapt.adapt_storage(adaptor::ROCArrayAdaptor, xs::AbstractArray)
    new_parameters = (type_parameters(xs, (eltype, ndims))..., storagemode(adaptor))
    roctype = set_type_parameters(ROCArray, (eltype, ndims, storagemode), new_parameters)
    return isbits(xs) ? xs : adapt(roctype, xs)
end

function NDTensors.adapt_storagetype(
        adaptor::ROCArrayAdaptor, ::Type{EmptyStorage{ElT, StoreT}}
    ) where {ElT, StoreT}
    roctype = set_type_parameters(
        ROCVector, (eltype, storagemode), (ElT, storagemode(adaptor))
    )
    return emptytype(adapt_storagetype(roctype, StoreT))
end
