@traitfn function leaf_parenttype(
  arraytype::Type{ArrayT}
) where {ArrayT; IsWrappedArray{ArrayT}}
  return leaf_parenttype(parenttype(arraytype))
end

@traitfn function leaf_parenttype(
  arraytype::Type{ArrayT}
) where {ArrayT; !IsWrappedArray{ArrayT}}
  return arraytype
end

# For working with instances.
leaf_parenttype(array::AbstractArray) = leaf_parenttype(typeof(array))
