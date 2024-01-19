NDTensors.cpu(e::Exposed{<:MtlArray}) = adapt(Array, e)

function mtl(xs; storage=DefaultStorageMode)
  return adapt(set_storagemode(MtlArray, storage), xs)
end

# More general than the version in Metal.jl
function Adapt.adapt_storage(arraytype::Type{<:MtlArray}, xs::AbstractArray)
  # TODO: Do this in one call.
  arraytype_specified = specify_parameters(
    arraytype, get_parameter(xs, Position(1)), get_parameter(xs, Position(2)), get_parameter(xs, Position(3))
  )
  return isbitstype(typeof(xs)) ? xs : convert(arraytype_specified, xs)
end
