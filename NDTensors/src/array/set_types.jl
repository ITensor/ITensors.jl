using .TypeParameterAccessors: TypeParameterAccessors

TypeParameterAccessors.eltype_position(::Type{<:SubArray}) = 1

# TODO: Figure out how to define this properly.
# function set_ndims(arraytype::Type{<:SubArray}, ndims)
#   arraytype_1 = set_parameters(arraytype, Position(2), ndims)
#   parent_arraytype = get_parameter(arraytype, Position(3))
#   parent_arraytype_1 = set_ndims(parent_arraytype, ndims)
#   return set_parameters(arraytype_1, Position(3), parent_arraytype_1)
# end
