"""
    set_parameters(type::DataType, parameters::Tuple)

Set the parameters of DataType `type` to the tuple `paramters`.
"""
function set_parameters(type::DataType, parameters::Tuple)
  return unspecify_parameters(type){parameters...}
end

"""
    set_parameters(type::UnionAll, parameters::Tuple)

Set the parameters of UnionAll `type` to `paramters`.
"""
function set_parameters(type::UnionAll, parameters::Tuple)
  return to_unionall(set_parameters(to_datatype(type), parameters), type)
end

"""
    set_parameters(type)

If set_parameters is called with no `parameters` it returns the input unchanged.
"""
set_parameters(type) = type

"""
    set_parameter(type::Type, pos::Int, val)

Set the parameter of the Type `type` in the position `pos` with the value `val`
"""
Base.@assume_effects :foldable function set_parameter(type::Type, pos::Int, val)
  params = Base.unwrap_unionall(type).parameters
  return Base.rewrap_unionall(
    Base.typename(type).wrapper{params[1:(pos - 1)]...,val,params[(pos + 1):end]...}, type
  )
end

"""
    set_parameter(type::Type, val)

Set the parameter of the Type `type` in the first position with the value `val`. Note, this function is for types which only have a single parameter.
"""
set_parameter(type::Type, val) = set_parameters(type, tuple(val))

"""
    set_parameter(type::Type, fun::Function, parameter)

Set the parameter with the function tag `fun` to `parameter` for the Type `type` 
"""
set_parameter(type::Type, fun::Function, parameter::Type) = set_parameter(type, position(type, fun), parameter)

set_parameter(type::Type, fun::Function, parameter) = set_parameter(type, position(type, fun), TypeParameter(parameter))
