is_specified_parameter(param::TypeParameter) = true
is_specified_parameter(param::UnspecifiedTypeParameter) = false
function is_parameter_specified(type::Type, pos)
  return is_specified_parameter(wrapped_type_parameter(type, pos))
end
