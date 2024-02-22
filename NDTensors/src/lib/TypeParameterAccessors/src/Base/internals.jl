# Relies on Julia internals
@generated to_datatype(::Type{type}) where {type} = Base.unwrap_unionall(type)
#to_datatype(type::DataType) = type

"""
    parameters(type::DataType)

Gets all the type parameters of the DataType `type`.
"""
parameters(type::DataType) = Tuple(type.parameters)

@generated to_unionall(type::Type, ::Type{ref_type}) where {ref_type} =
  Base.rewrap_unionall(parameter(type), ref_type)
@generated unspecify_parameters(::Type{type}) where {type} = Base.typename(type).wrapper
