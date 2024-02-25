is_specified_parameter(param::TypeParameter) = true
is_specified_parameter(param::UnspecifiedTypeParameter) = false
function is_parameter_specified(type::Type, pos)
  return is_specified_parameter(AbstractTypeParameter(type, pos))
end
