function mtl(xs; storage=DefaultStorageMode)
  return adapt(set_storagemode(MtlArray, storage), xs)
end

# More general than the version in Metal.jl
function Adapt.adapt_storage(arraytype::Type{<:MtlArray}, xs::AbstractArray)
  # TODO: Do this in one call.
  arraytype_specified_1 = set_unspecified_parameters(
    arraytype, Position(1), get_parameter(xs, Position(1))
  )
  arraytype_specified_2 = set_unspecified_parameters(
    arraytype_specified_1, Position(2), get_parameter(xs, Position(2))
  )
  arraytype_specified_3 = set_unspecified_parameters(
    arraytype_specified_2, Position(3), get_parameter(xs, Position(3))
  )
  return isbitstype(typeof(xs)) ? xs : convert(arraytype_specified_3, xs)
end
