using SimpleTraits: SimpleTraits, @traitfn

# Required overload implementation
position(::Type{<:AbstractArray}, ::typeof(eltype)) = Position(1)
position(::Type{<:AbstractArray}, ::typeof(ndims)) = Position(2)

@traitfn function set_eltype(
  type::Type{ArrayT}, elt::Type
) where {ArrayT <: AbstractArray; !IsWrappedArray{ArrayT}}
  return set_parameter(type, eltype, elt)
end

@traitfn function set_ndims(
  type::Type{ArrayT}, ndim::Int
) where {ArrayT <: AbstractArray; !IsWrappedArray{ArrayT}}
  return set_parameter(type, ndims, ndim)
end
