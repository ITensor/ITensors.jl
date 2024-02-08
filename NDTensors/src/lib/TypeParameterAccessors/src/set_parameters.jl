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
function set_parameter(type::Type, pos::Int, val)
  params = parameters(type)
  new_params = Base.setindex(params, val, pos)
  return set_parameters(type, new_params)
end

"""
    set_parameter(type::Type, val)

Set the parameter of the Type `type` in the first position with the value `val`. Note, this function is for types which only have a single parameter.
"""
set_parameter(type::Type, val) = set_parameters(type, tuple(val))
