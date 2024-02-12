## TODO still working on this
function specify_parameter(type::Type, pos::Int, param::Type)
  is_parameter_specified(type, pos) && return type
  return set_parameter(type, Position(pos), param)
end

function specify_parameter(type::Type, pos::Int, param)
  is_parameter_specified(type, pos) && return type
  return set_parameter(type, Position(pos), TypeParameter(param))
end

"""
Set parameters starting from the first one if they are unspecified.
"""
## TODO This is not a type stable definition because of an issue in `Julia` see if there is a better way to do this.
function specify_parameters(type::Type, parameter...)
  params = tuple(parameter...)
  for i in 1:length(params)
    type = specify_parameter(type, i, params[i])
  end
  return type
end

# # Base case, set the type parameter at the position if it is unspecified.
# function specify_parameters(type::Type, position::Position, parameter)
#   current_parameter = get_parameter(type, position)
#   return replace_unspecified_parameters(type, position, current_parameter, parameter)
# end

# function replace_unspecified_parameters(
#   type::Type, position::Position, current_parameter::Type{<:UnspecifiedParameter}, parameter
# )
#   return set_parameters(type, position, parameter)
# end

# function replace_unspecified_parameters(
#   type::Type, position::Position, current_parameter, parameter
# )
#   return type
# end

# # Implementation in terms of generic version.
# function specify_parameters(type::Type, position::Position, parameters...)
#   return set_parameters(specify_parameters, type, position, parameters...)
# end

# function specify_parameters(type::Type)
#   return specify_parameters(type, DefaultParameters())
# end

# function specify_parameter(type::Type, Position::Position, parameter)
#   return specify_parameters(type, Position, parameter)
# end
