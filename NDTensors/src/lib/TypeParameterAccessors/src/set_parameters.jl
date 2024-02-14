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
Base.@assume_effects :foldable function set_parameters(type::UnionAll, parameters::Tuple)
  return Base.rewrap_unionall(set_parameters(Base.unwrap_unionall(type), parameters), type)
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
    set_parameter(::Type{Typ}, ::Position{Pos}, ::Type{Param})

Sets the parameter at Position `Pos` of the Type `Typ` to the new Type `param`.
This function is necessary to ensure type stability.
"""
@generated function set_parameter(
  ::Type{Typ}, ::Position{Pos}, ::Type{Param}
) where {Typ,Pos,Param}
  return set_parameter(Typ, Pos, Param)
end

"""
    set_parameter(::Type{Typ}, ::Position{Pos}, ::TypeParameter{Param})

Sets the parameter at Position `Pos` of the Type `Typ` to the new Type `param`.
This function is necessary to ensure type stability.
"""
@generated function set_parameter(
  ::Type{Typ}, ::Position{Pos}, ::TypeParameter{Param}
) where {Typ,Pos,Param}
  return set_parameter(Typ, Pos, Param)
end

set_parameter(type::Type{Typ}, ::UndefinedPosition, val) where {Typ} = type

"""
    set_parameter(type::Type, fun::Function, parameter)

Set the parameter with the function tag `fun` to `parameter` for the Type `type` 
"""
set_parameter(type::Type, fun::Function, parameter::Type) =
  set_parameter(type, position(type, fun), parameter)

function set_parameter(type::Type, fun::Function, parameter)
  return set_parameter(type, position(type, fun), TypeParameter(parameter))
end
