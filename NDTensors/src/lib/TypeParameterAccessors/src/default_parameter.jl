"""
Get the default parameter of an object which is associated with a specific function tag.
"""
default_parameter(object, fun::typeof(Function)) = default_parameter(typeof(object), fun)

default_parameters(::Type)::Tuple = ()

default_parameters(::Type{<:AbstractArray}) = (eltype, ndims)
