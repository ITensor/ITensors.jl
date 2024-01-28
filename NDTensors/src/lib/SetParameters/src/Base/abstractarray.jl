using SimpleTraits: SimpleTraits, @traitfn
using NDTensors.Unwrap: IsWrappedArray, parenttype

@traitfn function set_eltype(
  type::Type{ArrayT}, elt::Type
) where {ArrayT <: AbstractArray; !IsWrappedArray{ArrayT}}
  return set_parameter(type, eltype_position(type), elt)
end

@traitfn function set_eltype(
  type::Type{ArrayT}, elt::Type
) where {ArrayT <: AbstractArray; IsWrappedArray{ArrayT}}
  new_parenttype = set_eltype(parenttype(type), elt)
  return set_parenttype(set_parameter(type, eltype_position(type), elt), new_parenttype)
end

function set_parenttype(t::Type, parent_type)
  return set_parameter(t, parenttype_position(t), parent_type)
end

eltype_position(::Type{<:AbstractArray}) = Position(1)
## This will fail for some wrapped types so potentially set for array and other types?
ndims_position(::Type{<:AbstractArray}) = Position(2)

@traitfn function set_ndims(
  type::Type{ArrayT}, ndim::Int
) where {ArrayT <: AbstractArray; !IsWrappedArray{ArrayT}}
  return set_parameter(type, ndims_position(type), ndim)
end
