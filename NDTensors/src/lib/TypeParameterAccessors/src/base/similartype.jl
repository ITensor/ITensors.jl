"""
`set_indstype` should be overloaded for
types with structured dimensions,
like `OffsetArrays` or named indices
(such as ITensors).
"""
function set_indstype(arraytype::Type{<:AbstractArray}, dims::Tuple)
  return set_ndims(arraytype, length(dims))
end

function similartype(arraytype::Type{<:AbstractArray}, eltype::Type, dims::Tuple)
  return similartype(similartype(arraytype, eltype), dims)
end

@traitfn function similartype(
  arraytype::Type{ArrayT}, eltype::Type
) where {ArrayT; !IsWrappedArray{ArrayT}}
  return set_eltype(arraytype, eltype)
end

@traitfn function similartype(
  arraytype::Type{ArrayT}, dims::Tuple
) where {ArrayT; !IsWrappedArray{ArrayT}}
  return set_indstype(arraytype, dims)
end

function similartype(arraytype::Type{<:AbstractArray}, dims::Base.DimOrInd...)
  return similartype(arraytype, dims)
end

function similartype(arraytype::Type{<:AbstractArray})
  return similartype(arraytype, eltype(arraytype))
end

## Wrapped arrays
@traitfn function similartype(
  arraytype::Type{ArrayT}, eltype::Type
) where {ArrayT; IsWrappedArray{ArrayT}}
  return similartype(unwrap_array_type(arraytype), eltype)
end

@traitfn function similartype(
  arraytype::Type{ArrayT}, dims::Tuple
) where {ArrayT; IsWrappedArray{ArrayT}}
  return similartype(unwrap_array_type(arraytype), dims)
end

# This is for uniform `Diag` storage which uses
# a Number as the data type.
# TODO: Delete this when we change to using a
# `FillArray` instead. This is a stand-in
# to make things work with the current design.
function similartype(numbertype::Type{<:Number})
  return numbertype
end

# Instances
function similartype(array::AbstractArray, eltype::Type, dims...)
  return similartype(typeof(array), eltype, dims...)
end
similartype(array::AbstractArray, eltype::Type) = similartype(typeof(array), eltype)
similartype(array::AbstractArray, dims...) = similartype(typeof(array), dims...)
