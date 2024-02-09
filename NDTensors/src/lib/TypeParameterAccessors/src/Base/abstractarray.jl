using SimpleTraits

eltype_position(::Type{<:AbstractArray}) = 1
## This will fail for some wrapped types so potentially set for array and other types?
ndims_position(::Type{<:AbstractArray}) = 2

@traitfn function set_eltype(
  type::Type{ArrayT}, elt::Type
) where {ArrayT <: AbstractArray; !IsWrappedArray{ArrayT}}
  return set_parameter(type, eltype_position(type), elt)
end

@traitfn function set_ndims(
  type::Type{ArrayT}, ndim::Int
) where {ArrayT <: AbstractArray; !IsWrappedArray{ArrayT}}
  return set_parameter(type, ndims_position(type), ndim)
end
