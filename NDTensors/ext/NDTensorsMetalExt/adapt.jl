using NDTensors.MetalExtensions: MetalExtensions
using NDTensors.GPUArraysCoreExtensions: GPUArraysCoreExtensions, set_storagemode
using NDTensors.TypeParameterAccessors: specify_type_parameters, type_parameters

GPUArraysCoreExtensions.cpu(e::Exposed{<:MtlArray}) = adapt(Array, e)

function MetalExtensions.mtl(xs; storage=DefaultStorageMode)
  return adapt(set_storagemode(MtlArray, storage), xs)
end

# More general than the version in Metal.jl
## TODO Rewrite this using a custom `MtlArrayAdaptor` which will be written in  `MetalExtensions`.
function Adapt.adapt_storage(arraytype::Type{<:MtlArray}, xs::AbstractArray)
  params = type_parameters(xs)
  arraytype_specified = specify_type_parameters(arraytype, params)
  return isbitstype(typeof(xs)) ? xs : convert(arraytype_specified, xs)
end
