using NDTensors.TypeParameterAccessors: IsWrappedArray, parenttype
using SimpleTraits: SimpleTraits, @traitfn
## These functions will be used in place of unwrap_array_type but will be
## call indirectly through the expose function.
@traitfn function unwrap_array_type(
  arraytype::Type{ArrayT}
) where {ArrayT; IsWrappedArray{ArrayT}}
  return unwrap_array_type(parenttype(arraytype))
end

@traitfn function unwrap_array_type(
  arraytype::Type{ArrayT}
) where {ArrayT; !IsWrappedArray{ArrayT}}
  return arraytype
end

# For working with instances.
unwrap_array_type(array::AbstractArray) = unwrap_array_type(typeof(array))
unwrap_array_type(E::Exposed) = unwrap_array_type(unexpose(E))
