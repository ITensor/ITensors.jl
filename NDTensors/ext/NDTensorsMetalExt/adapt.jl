NDTensors.cpu(e::Exposed{<:MtlArray}) = adapt(Array, e)

function mtl(xs; storage=DefaultStorageMode)
  return adapt(set_storagemode(MtlArray, storage), xs)
end

# More general than the version in Metal.jl
function Adapt.adapt_storage(arraytype::Type{<:MtlArray}, xs::AbstractArray)
  ## TODO fix this too
  #arraytype_specified = specify_parameters(arraytype, (eltype, ndims, alloctype), parameters(xs)) 
  #arraytype_specified = specify_parameters(arraytype, parameters(xs))
  return isbitstype(typeof(xs)) ? xs : convert(arraytype_specified, xs)
end
