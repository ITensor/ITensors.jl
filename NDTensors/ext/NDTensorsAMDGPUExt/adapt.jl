using NDTensors: NDTensors, EmptyStorage, adapt_storagetype, emptytype
using NDTensors.AMDGPUExtensions: AMDGPUExtensions, ROCArrayAdaptor
using NDTensors.GPUArraysCoreExtensions: storagemode
using NDTensors.TypeParameterAccessors:
  set_type_parameter, set_type_parameters, type_parameter, type_parameters
using Adapt: Adapt, adapt
using AMDGPU: AMDGPU, ROCArray, ROCVector

function AMDGPUExtensions.roc(xs)
  return fmap(x -> adapt(ROCArrayAdaptor{AMDGPU.Runtime.Mem.HIPBuffer}(), x), xs)
end

function Adapt.adapt_storage(adaptor::ROCArrayAdaptor, xs::AbstractArray)
  roctype = set_type_parameters(
    ROCArray, (eltype, ndims), type_parameters(xs, (eltype, ndims))
  )
  roctype = set_type_parameter(roctype, storagemode, storagemode(adaptor))

  return isbits(xs) ? xs : adapt(roctype, xs)
end

function NDTensors.adapt_storagetype(
  adaptor::ROCArrayAdaptor, xs::Type{EmptyStorage{ElT,StoreT}}
) where {ElT,StoreT}
  roctype = set_type_parameters(
    ROCVector, (eltype, storagemode), (ElT, storagemode(adaptor))
  )
  return emptytype(adapt_storagetype(roctype, StoreT))
end
