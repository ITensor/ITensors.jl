"""
Get the default parameter of an object which is associated with a specific function tag.
"""
default_parameter(object, fun::Function) = default_parameter(typeof(object), fun)

Base.@assume_effects :foldable function default_parameters(type::Type)
  return Tuple(map(i -> parameter_function(type, Position(i)), 1:nparameters(type)))
end
