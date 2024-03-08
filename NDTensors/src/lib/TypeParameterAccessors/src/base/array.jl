position(::Type{<:Array}, ::typeof(eltype)) = Position(1)
position(::Type{<:Array}, ::typeof(ndims)) = Position(2)

default_type_parameters(::Type{<:Array}) = (Float64, 1)

function set_ndims(type::Type{<:Array}, param)
  return set_type_parameter(type, ndims, param)
end
