using NDTensors.TypeParameterAccessors: TypeParameterAccessors
NDTensors.cpu(e::Exposed{<:MtlArray}) = adapt(Array, e)

function mtl(xs; storage=DefaultStorageMode)
  return adapt(set_storagemode(MtlArray, storage), xs)
end

# More general than the version in Metal.jl
function Adapt.adapt_storage(arraytype::Type{<:MtlArray}, xs::AbstractArray)
  arraytype_specified = TypeParameterAccessors.specify_type_parameters(
    arraytype, (eltype, Base.ndims), TypeParameterAccessors.type_parameters(xs, (eltype, Base.ndims))
  )
  return isbitstype(typeof(xs)) ? xs : convert(arraytype_specified, xs)
end
