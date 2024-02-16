"""
Get the default parameter of an object which is associated with a specific function tag.
"""
default_parameter(object, fun::Function) = default_parameter(typeof(object), fun)

paramter_function(::Type{<:AbstractArray}) = (eltype, ndims)
