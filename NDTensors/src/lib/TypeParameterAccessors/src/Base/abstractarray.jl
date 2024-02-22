using SimpleTraits: SimpleTraits, @traitfn

# Required overload implementation
position(::Type{<:AbstractArray}, ::typeof(eltype)) = Position(1)
position(::Type{<:AbstractArray}, ::typeof(ndims)) = Position(2)

## TODO I don't think this is the right place for this but define it here for now
default_parameter(::Type{<:AbstractArray}, ::Position{1}) = Float64
default_parameter(::Type{<:AbstractArray}, ::Position{2}) = 1

parameter_name(::Type{<:AbstractArray}, ::Position{1}) = eltype
parameter_name(::Type{<:AbstractArray}, ::Position{2}) = ndims

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
