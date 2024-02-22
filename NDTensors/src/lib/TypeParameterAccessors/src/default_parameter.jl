using Compat: Compat
"""
Get the default parameter of an object which is associated with a specific function tag.
"""
default_parameter(object, fun::Function) = default_parameter(typeof(object), fun)

default_parameter(type::Type, fun::Function) = default_parameter(type, position(type, fun))

Compat.@assume_effects :foldable function default_parameters(type::Type)
  return map(name -> default_parameter(type, name), parameter_names(type))
end
