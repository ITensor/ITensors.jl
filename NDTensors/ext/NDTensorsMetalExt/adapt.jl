using NDTensors.MetalExtensions: MetalExtensions
using NDTensors.GPUArraysCoreExtensions: GPUArraysCoreExtensions

GPUArraysCoreExtensions.cpu(e::Exposed{<:MtlArray}) = adapt(Array, e)

function MetalExtensions.mtl(xs; storage=DefaultStorageMode)
  return adapt(set_storagemode(MtlArray, storage), xs)
end

# More general than the version in Metal.jl
function Adapt.adapt_storage(arraytype::Type{<:MtlArray}, xs::AbstractArray)
  params = get_parameters(xs)
  arraytype_specified = specify_parameters(arraytype, params...)
  return isbitstype(typeof(xs)) ? xs : convert(arraytype_specified, xs)
end
