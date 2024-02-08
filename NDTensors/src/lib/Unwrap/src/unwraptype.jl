using NDTensors.TypeParameterAccessors: IsWrappedArray, parenttype
using SimpleTraits: SimpleTraits, @traitfn
## These functions will be used in place of unwrap_type but will be
## call indirectly through the expose function.
@traitfn function unwrap_type(
  arraytype::Type{ArrayT}
) where {ArrayT; IsWrappedArray{ArrayT}}
  return unwrap_type(parenttype(arraytype))
end

@traitfn function unwrap_type(
  arraytype::Type{ArrayT}
) where {ArrayT; !IsWrappedArray{ArrayT}}
  return arraytype
end

# For working with instances.
unwrap_type(array::AbstractArray) = unwrap_type(typeof(array))
unwrap_type(E::Exposed) = unwrap_type(unexpose(E))
