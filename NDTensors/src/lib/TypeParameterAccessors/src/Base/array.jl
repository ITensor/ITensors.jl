position(::Type{<:Array}, ::typeof(eltype)) = Position(1)
position(::Type{<:Array}, ::typeof(ndims)) = Position(2)
position_name(::Type{<:Array}, ::Position{1}) = eltype
position_name(::Type{<:Array}, ::Position{2}) = ndims

default_parameter(::Type{<:Array}, ::typeof(eltype)) = Float64
default_parameter(::Type{<:Array}, ::typeof(ndims)) = 1

function set_ndims(type::Type{<:Array}, ndim::Int)
  return set_parameter(type, ndims, ndim)
end
