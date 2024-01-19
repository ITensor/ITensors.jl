function set_eltype(type::Type{<:AbstractArray}, elt::Type)
  return set_parameter(type, Position(1), elt)
end

function set_ndims(type::Type{<:AbstractArray}, ndim::Int)
  return set_parameter(type, Position(2), ndim)
end