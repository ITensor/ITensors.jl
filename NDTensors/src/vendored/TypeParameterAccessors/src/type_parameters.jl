"""
  struct Position{P} end

Singleton type to statically represent the type-parameter position.
This is meant for internal use as a `Val`-like structure to improve type-inference.
"""
struct Position{P} end
Position(pos::Int) = Position{pos}()

Base.Int(pos::Position) = Int(typeof(pos))
Base.Int(::Type{Position{P}}) where {P} = Int(P)
Base.to_index(pos::Position) = Base.to_index(typeof(pos))
Base.to_index(::Type{P}) where {P<:Position} = Int(P)

"""
  position(type::Type, position_name)::Position

An optional interface function. Defining this allows accessing a parameter
at the defined position using the `position_name`.

For example, defining `TypeParameterAccessors.position(::Type{<:MyType}, ::typeof(eltype)) = Position(1)`
allows accessing the first type parameter with `type_parameters(MyType(...), eltype)`,
in addition to the standard `type_parameters(MyType(...), 1)` or `type_parameters(MyType(...), Position(1))`.
"""
function position end

position(object, name) = position(typeof(object), name)
position(::Type, pos::Int) = Position(pos)
position(::Type, pos::Position) = pos

function position(type::Type, name)
  type′ = unspecify_type_parameters(type)
  if type === type′
    # Fallback definition that determines the
    # position automatically from the supertype of
    # the type.
    return position_from_supertype(type′, name)
  end
  return position(type′, name)
end

# Automatically determine the position of a type parameter of a type given
# a supertype and the name of the parameter.
function position_from_supertype(type::Type, name)
  type′ = unspecify_type_parameters(type)
  supertype_pos = position(supertype(type′), name)
  return position_from_supertype_position(type′, supertype_pos)
end

# Automatically determine the position of a type parameter of a type given
# the supertype and the position of the corresponding parameter in the supertype.
@generated function position_from_supertype_position(
  ::Type{T}, supertype_pos::Position
) where {T}
  T′ = unspecify_type_parameters(T)
  # The type parameters of the type as TypeVars.
  # TODO: Ideally we would use `get_type_parameters`
  # but that sometimes loses TypeVar names:
  # https://github.com/ITensor/TypeParameterAccessors.jl/issues/30
  type_params = Base.unwrap_unionall(T′).parameters
  # The type parameters of the immediate supertype as TypeVars.
  # This has TypeVars with names that correspond to the names of
  # the TypeVars of the type parameters of `T`, for example:
  # ```julia
  # julia> struct MyArray{B,A} <: AbstractArray{A,B} end
  #
  # julia> Base.unwrap_unionall(MyArray).parameters
  # svec(B, A)
  #
  # julia> Base.unwrap_unionall(supertype(MyArray)).parameters
  # svec(A, B)
  # ```
  supertype_params = Base.unwrap_unionall(supertype(T′)).parameters
  supertype_param = supertype_params[Int(supertype_pos)]
  if !(supertype_param isa TypeVar)
    error("Position not found.")
  end
  pos = findfirst(param -> (param.name == supertype_param.name), type_params)
  if isnothing(pos)
    return error("Position not found.")
  end
  return :(@inline; $(Position(pos)))
end

function positions(::Type{T}, pos::Tuple) where {T}
  return ntuple(length(pos)) do i
    return position(T, pos[i])
  end
end

"""
  get_type_parameters(type_or_obj, [pos])

Return a tuple containing the type parameters of a given type or object.
Optionally you can specify a position to just get the parameter for that position,
or a tuple of positions to get a subset of parameters.

If parameters are unspecified, returns a `TypeVar`. For a checked version,
see [`type_parameters`](@ref).
"""
function get_type_parameters end

# This implementation is type-stable in 1.11, but not in 1.10.
# Attempts with `Base.@constprop :aggressive` failed, so generated function instead
# @inline type_parameters(::Type{T}) where {T} = Tuple(Base.unwrap_unionall(T).parameters)
@generated function get_type_parameters(::Type{T}) where {T}
  params = wrap_symbol_quotenode.(Tuple(Base.unwrap_unionall(T).parameters))
  return :(@inline; ($(params...),))
end
@inline get_type_parameters(::Type{T}, pos) where {T} = get_type_parameters(
  T, position(T, pos)
)
@inline get_type_parameters(::Type{T}, ::Position{p}) where {T,p} = get_type_parameters(T)[p]
@inline get_type_parameters(::Type{T}, ::Position{0}) where {T} = T
@inline get_type_parameters(::Type{T}, pos::Tuple) where {T} = get_type_parameters.(T, pos)
@inline get_type_parameters(object, pos) = get_type_parameters(typeof(object), pos)
@inline get_type_parameters(object) = get_type_parameters(typeof(object))

"""
  type_parameters(type_or_obj, [pos])

Return a tuple containing the type parameters of a given type or object.
Optionally you can specify a position to just get the parameter for that position,
or a tuple of positions to get a subset of parameters.

Errors if parameters are unspecified. For an unchecked version,
see [`get_type_parameters`](@ref).
"""
function type_parameters end

function type_parameters(::Type{T}) where {T}
  params = get_type_parameters(T)
  any(param -> param isa TypeVar, params) &&
    return error("One or more parameter is not specified.")
  return params
end
@inline function type_parameters(::Type{T}, pos) where {T}
  param = get_type_parameters(T, pos)
  param isa TypeVar && return error("The parameter is not specified.")
  return param
end
@inline type_parameters(::Type{T}, pos::Tuple) where {T} = type_parameters.(T, pos)
@inline type_parameters(object, pos) = type_parameters(typeof(object), pos)
@inline type_parameters(object) = type_parameters(typeof(object))

"""
  nparameters(type_or_obj)

Return the number of type parameters for a given type or object.
"""
nparameters(object) = nparameters(typeof(object))
nparameters(::Type{T}) where {T} = length(get_type_parameters(T))

"""
  is_parameter_specified(type::Type, pos)

Return whether or not the type parameter at a given position is considered specified.
"""
function is_parameter_specified(::Type{T}, pos) where {T}
  return !(get_type_parameters(T, pos) isa TypeVar)
end

"""
  unspecify_type_parameters(type::Type, [positions::Tuple])
  unspecify_type_parameters(type::Type, position)

Return a new type where the type parameters at the given positions are unset.
"""
unspecify_type_parameters(::Type{T}) where {T} = Base.typename(T).wrapper
function unspecify_type_parameters(::Type{T}, pos::Tuple) where {T}
  @inline
  return unspecify_type_parameters(T, positions(T, pos))
end
@generated function unspecify_type_parameters(
  ::Type{T}, positions::Tuple{Vararg{Position}}
) where {T}
  allparams = collect(Any, get_type_parameters(T))
  for pos in type_parameters(positions)
    allparams[pos] = get_type_parameters(unspecify_type_parameters(T), Int(pos))
  end
  type_expr = construct_type_expr(T, allparams)
  return :(@inline; $type_expr)
end
unspecify_type_parameters(::Type{T}, pos) where {T} = unspecify_type_parameters(T, (pos,))

"""
  set_type_parameters(type::Type, positions::Tuple, parameters::Tuple)
  set_type_parameters(type::Type, position, parameter)

Return a new type where the type parameters at the given positions are set to the provided values.
"""
function set_type_parameters(
  ::Type{T}, pos::Tuple{Vararg{Any,N}}, parameters::Tuple{Vararg{Any,N}}
) where {T,N}
  return set_type_parameters(T, positions(T, pos), parameters)
end
@generated function set_type_parameters(
  ::Type{T}, positions::Tuple{Vararg{Position,N}}, params::Tuple{Vararg{Any,N}}
) where {T,N}
  # collect parameters and change
  allparams = collect(Any, get_type_parameters(T))
  for (i, pos) in enumerate(get_type_parameters(positions))
    allparams[pos] = :(params[$i])
  end
  type_expr = construct_type_expr(T, allparams)
  return :(@inline; $type_expr)
end
function set_type_parameters(::Type{T}, pos, param) where {T}
  return set_type_parameters(T, (pos,), (param,))
end

"""
  specify_type_parameters(type::Type, positions::Tuple, parameters::Tuple)
  specify_type_parameters(type::Type, position, parameter)

Return a new type where the type parameters at the given positions are set to the provided values,
only if they were previously unspecified.
"""
function specify_type_parameters(
  ::Type{T}, pos::Tuple{Vararg{Any,N}}, parameters::Tuple{Vararg{Any,N}}
) where {T,N}
  return specify_type_parameters(T, positions(T, pos), parameters)
end
function specify_type_parameters(::Type{T}, parameters::Tuple) where {T}
  return specify_type_parameters(T, ntuple(identity, nparameters(T)), parameters)
end
@generated function specify_type_parameters(
  ::Type{T}, positions::Tuple{Vararg{Position,N}}, params::Tuple{Vararg{Any,N}}
) where {T,N}
  # collect parameters and change unspecified
  allparams = collect(Any, get_type_parameters(T))
  for (i, pos) in enumerate(get_type_parameters(positions))
    if !is_parameter_specified(T, pos())
      allparams[pos] = :(params[$i])
    end
  end
  type_expr = construct_type_expr(T, allparams)
  return :(@inline; $type_expr)
end
function specify_type_parameters(::Type{T}, pos, param) where {T}
  return specify_type_parameters(T, (pos,), (param,))
end

"""
  default_type_parameters(type::Type)::Tuple

An optional interface function. Defining this allows filling type parameters
of the specified type with default values.

This function should output a Tuple of the default values, with exactly
one for each type parameter slot of the type.
"""
function default_type_parameters(::Type{T}, pos) where {T}
  return default_type_parameters(T, position(T, pos))
end
function default_type_parameters(::Type{T}, ::Position{pos}) where {T,pos}
  param = default_type_parameters(T)[pos]
  if param isa UndefinedDefaultTypeParameter
    return error("No default parameter defined at this position.")
  end
  return param
end
default_type_parameters(::Type{T}, pos::Tuple) where {T} = default_type_parameters.(T, pos)
default_type_parameters(t, pos) = default_type_parameters(typeof(t), pos)
default_type_parameters(t) = default_type_parameters(typeof(t))
function default_type_parameters(type::Type)
  type′ = unspecify_type_parameters(type)
  if type === type′
    return default_type_parameters_from_supertype(type′)
  end
  return default_type_parameters(type′)
end

struct UndefinedDefaultTypeParameter end

# Determine the default type parameters of a type from the default type
# parameters of the supertype of the type. Uses similar logic as
# `position_from_supertype_position` for matching TypeVar names
# between the type and the supertype. Type parameters that exist
# in the type but not the supertype will have a default type parameter
# `UndefinedDefaultTypeParameter()`. Accessing those type parameters
# by name/position will throw an error.
@generated function default_type_parameters_from_supertype(::Type{T}) where {T}
  T′ = unspecify_type_parameters(T)
  supertype_default_type_params = default_type_parameters(supertype(T′))
  type_params = Base.unwrap_unionall(T′).parameters
  supertype_params = Base.unwrap_unionall(supertype(T′)).parameters
  defaults = Any[UndefinedDefaultTypeParameter() for _ in 1:nparameters(T′)]
  for (supertype_param, supertype_default_type_param) in
      zip(supertype_params, supertype_default_type_params)
    if !(supertype_param isa TypeVar)
      continue
    end
    param_position = findfirst(param -> (param.name == supertype_param.name), type_params)
    defaults[param_position] = supertype_default_type_param
  end
  return :(@inline; $(Tuple(defaults)))
end

"""
  set_default_type_parameters(type::Type, [positions::Tuple])
  set_default_type_parameters(type::Type, position)

Set the type parameters at the given positions to their default values.
"""
function set_default_type_parameters(::Type{T}, pos::Tuple) where {T}
  return set_type_parameters(T, pos, default_type_parameters.(T, pos))
end
function set_default_type_parameters(::Type{T}) where {T}
  return set_default_type_parameters(T, ntuple(identity, nparameters(T)))
end
function set_default_type_parameters(::Type{T}, pos) where {T}
  return set_default_type_parameters(T, (pos,))
end

"""
  specify_default_type_parameters(type::Type, [positions::Tuple])
  specify_default_type_parameters(type::Type, position)

Set the type parameters at the given positions to their default values, if they
had not been specified.
"""
function specify_default_type_parameters(::Type{T}, pos::Tuple) where {T}
  return specify_type_parameters(T, pos, default_type_parameters.(T, pos))
end
function specify_default_type_parameters(::Type{T}) where {T}
  return specify_default_type_parameters(T, ntuple(identity, nparameters(T)))
end
function specify_default_type_parameters(::Type{T}, pos) where {T}
  return specify_default_type_parameters(T, (pos,))
end
