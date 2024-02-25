abstract type AbstractPosition end

struct Position{Pos} <: AbstractPosition end
Position(pos) = Position{pos}()
Base.Int(pos::Position) = Int(parameter(typeof(pos)))

struct UndefinedPosition <: AbstractPosition end

abstract type AbstractTypeParameter end

struct TypeParameter{Param} <: AbstractTypeParameter end
TypeParameter(param) = TypeParameter{param}()
TypeParameter(param::TypeParameter) = param
AbstractTypeParameter(param::AbstractTypeParameter) = param
AbstractTypeParameter(param) = TypeParameter(param)
AbstractTypeParameter(param::TypeVar) = UnspecifiedTypeParameter()
AbstractTypeParameter(type::Type, pos) = AbstractTypeParameter(parameter(type, pos))

struct UnspecifiedTypeParameter <: AbstractTypeParameter end

# Similar to `Base.rewrap_unionall` but handles
# more general cases of `TypeVar` parameters.
@generated function to_unionall(type_type::Type)
  type = only(type_type.parameters)
  params = Base.unwrap_unionall(type).parameters
  for i in reverse(eachindex(params))
    param = params[i]
    if param isa TypeVar
      type = UnionAll(param, type)
    end
  end
  return type
end

# The signature `parameters(::Type{type}) where {type}`
# doesn't work if `type` is a `DataType` with `TypeVar`s.
function _parameters(type::Type)
  return Tuple(Base.unwrap_unionall(type).parameters)
end
@generated function parameters(type_type::Type)
  type = only(type_type.parameters)
  return _parameters(type)
end
parameters(object) = parameters(typeof(object))

# These definitions help with generic code, where
# we don't know what kind of position will be passed
# but we want to canonicalize to `Position` positions.
position(type::Type, pos::Position) = pos
position(type::Type, pos::Int) = Position(pos)
# Used for named positions.
function position(type::Type, pos)
  return error(
    "Type parameter position not defined for type `$(type)` and position name `$(pos)`."
  )
end

parameter(type::Type, pos) = parameter(type, position(type, pos))
parameter(type::Type, pos::Position) = parameters(type)[Int(pos)]
parameter(type::Type, pos::Int) = parameter(type, Position(pos))
parameter(type::Type) = only(parameters(type))

is_specified_parameter(param::TypeParameter) = true
is_specified_parameter(param::UnspecifiedTypeParameter) = false
function is_parameter_specified(type::Type, pos)
  return is_specified_parameter(AbstractTypeParameter(type, pos))
end

function _unspecify_parameters(type::Type)
  return Base.typename(type).wrapper
end
@generated function unspecify_parameters(type_type::Type)
  type = parameter(type_type)
  return _unspecify_parameters(type)
end

# Like `set_parameters` but less strict, i.e. it allows
# setting with `TypeVar` while `set_parameters` would error.
function new_parameters(type::Type, params)
  return to_unionall(unspecify_parameters(type){params...})
end

function _unspecify_parameter(type::Type, pos::Int)
  !is_parameter_specified(type, pos) && return type
  unspecified_param = parameter(unspecify_parameters(type), pos)
  params = Base.setindex(parameters(type), unspecified_param, pos)
  return new_parameters(type, params)
end
@generated function unspecify_parameter(type_type::Type, pos_type::Position)
  type = parameter(type_type)
  pos = parameter(pos_type)
  return _unspecify_parameter(type, pos)
end
function unspecify_parameter(type::Type, pos)
  return unspecify_parameter(type, position(type, pos))
end

function _unspecify_parameters(type::Type, positions::Tuple{Vararg{Int}})
  for pos in positions
    type = unspecify_parameter(type, pos)
  end
  return type
end
@generated function unspecify_parameters(
  type_type::Type, positions_type::Tuple{Vararg{Position}}
)
  type = parameter(type_type)
  positions = parameter.(parameters(positions_type))
  return _unspecify_parameters(type, positions)
end
function unspecify_parameters(type::Type, positions::Tuple)
  return unspecify_parameters(type, position.(type, positions))
end

function _set_parameter(type::Type, pos::Int, param)
  params = Base.setindex(parameters(type), param, pos)
  return new_parameters(type, params)
end
@generated function set_parameter(
  type_type::Type, pos_type::Position, param_type::TypeParameter
)
  type = parameter(type_type)
  pos = parameter(pos_type)
  param = parameter(param_type)
  return _set_parameter(type, pos, param)
end
function set_parameter(type::Type, pos, param)
  return set_parameter(type, position(type, pos), param)
end
function set_parameter(type::Type, pos::Position, param)
  return set_parameter(type, pos, TypeParameter(param))
end
function set_parameter(type::Type, pos::Position, param::UnspecifiedTypeParameter)
  return unspecify_parameter(type, pos)
end

function specify_parameter(type::Type, pos, param)
  is_parameter_specified(type, pos) && return type
  return set_parameter(type, pos, param)
end

nparameters(type_or_object) = length(parameters(type_or_object))

eachposition(type_or_object) = ntuple(Position, Val(nparameters(type_or_object)))

type_parameter(param::TypeParameter) = parameter(typeof(param))
function type_parameter(param::UnspecifiedTypeParameter)
  return error("The requested type parameter isn't specified.")
end
function type_parameter(type::Type, pos)
  return type_parameter(AbstractTypeParameter(type, pos))
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

for f in [:set_parameter, :specify_parameter]
  fs = Symbol(f, :s)
  _fs = Symbol(:_, f, :s)
  @eval begin
    function $_fs(type::Type, positions::Tuple{Vararg{Int}}, params::Tuple)
      @assert length(positions) == length(params)
      for i in 1:length(positions)
        type = $f(type, positions[i], params[i])
      end
      return type
    end
    @generated function $fs(
      type_type::Type,
      positions_type::Tuple{Vararg{Position}},
      params_type::Tuple{Vararg{TypeParameter}},
    )
      type = parameter(type_type)
      positions = parameter.(parameters(positions_type))
      params = parameter.(parameters(params_type))
      return $_fs(type, positions, params)
    end
    function $fs(type::Type, positions::Tuple, params::Tuple)
      return $fs(type, position.(type, positions), TypeParameter.(params))
    end
    function $fs(type::Type, params::Tuple)
      return $fs(type, eachposition(type), params)
    end
  end
end

function position_name(type::Type, pos::Position)
  return UndefinedPosition()
end

function default_type_parameter(type::Type, pos::Position)
  return default_type_parameter(type, position_name(type, pos))
end
default_type_parameter(type::Type, pos) = default_type_parameter(type, position(type, pos))
function default_type_parameter(type::Type, pos::UndefinedPosition)
  return UnspecifiedTypeParameter()
end
default_type_parameter(object, pos) = default_type_parameter(typeof(object), pos)

function default_type_parameters(type_or_object, positions::Tuple{Vararg{Position}})
  return map(pos -> default_type_parameter(type_or_object, pos), positions)
end
function default_type_parameters(
  type_or_object, positions::Tuple=eachposition(type_or_object)
)
  return default_type_parameters(type_or_object, position.(type_or_object, positions))
end

function set_default_parameter(type::Type, pos)
  return set_parameter(type, pos, default_type_parameter(type, pos))
end

function set_default_parameters(type::Type, positions::Tuple=eachposition(type))
  return set_parameters(type, positions, default_type_parameters(type, positions))
end

function specify_default_parameter(type::Type, pos)
  return specify_parameter(type, pos, default_type_parameter(type, pos))
end

function specify_default_parameters(type::Type, positions::Tuple=eachposition(type))
  return specify_parameters(type, positions, default_type_parameters(type, positions))
end
