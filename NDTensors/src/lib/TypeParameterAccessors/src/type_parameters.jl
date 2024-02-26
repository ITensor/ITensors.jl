type_parameter(param::TypeParameter) = parameter(typeof(param))
function type_parameter(param::UnspecifiedTypeParameter)
  return error("The requested type parameter isn't specified.")
end
function type_parameter(type::Type, pos)
  return type_parameter(wrapped_type_parameter(type, pos))
end
function type_parameter(object, pos)
  return type_parameter(typeof(object), pos)
end
function type_parameter(type_or_object)
  return only(type_parameters(type_or_object))
end

function type_parameters(type_or_object, positions=eachposition(type_or_object))
  return map(pos -> type_parameter(type_or_object, pos), positions)
end
