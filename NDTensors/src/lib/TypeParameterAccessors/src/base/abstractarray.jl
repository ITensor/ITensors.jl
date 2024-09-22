struct Self end

parameter(type::Type, pos::Self) = type
function set_type_parameter(type::Type, pos::Self, param)
  return error("Can't set the parent type of an unwrapped array type.")
end

function set_eltype(array::AbstractArray, param)
  return convert(set_eltype(typeof(array), param), array)
end

## This will fail if position of `ndims` is not defined for `type`
function set_ndims(type::Type{<:AbstractArray}, param)
  return set_type_parameter(type, ndims, param)
end

using SimpleTraits: SimpleTraits, @traitdef, @traitimpl

# Trait indicating if the AbstractArray type is an array wrapper.
# Assumes that it implements `NDTensors.parenttype`.
@traitdef IsWrappedArray{ArrayType}

#! format: off
@traitimpl IsWrappedArray{ArrayType} <- is_wrapped_array(ArrayType)
#! format: on

parenttype(type::Type{<:AbstractArray}) = parameter(type, parenttype)
parenttype(object::AbstractArray) = parenttype(typeof(object))
position(::Type{<:AbstractArray}, ::typeof(parenttype)) = Self()

is_wrapped_array(arraytype::Type{<:AbstractArray}) = (parenttype(arraytype) ≠ arraytype)
@inline is_wrapped_array(array::AbstractArray) = is_wrapped_array(typeof(array))
@inline is_wrapped_array(object) = false

using SimpleTraits: Not, @traitfn

@traitfn function unwrap_array_type(
  arraytype::Type{ArrayType}
) where {ArrayType; IsWrappedArray{ArrayType}}
  return unwrap_array_type(parenttype(arraytype))
end

@traitfn function unwrap_array_type(
  arraytype::Type{ArrayType}
) where {ArrayType; !IsWrappedArray{ArrayType}}
  return arraytype
end

# For working with instances.
unwrap_array_type(array::AbstractArray) = unwrap_array_type(typeof(array))

function set_parenttype(t::Type, param)
  return set_type_parameter(t, parenttype, param)
end

@traitfn function set_eltype(
  type::Type{ArrayType}, param
) where {ArrayType <: AbstractArray; IsWrappedArray{ArrayType}}
  new_parenttype = set_eltype(parenttype(type), param)
  # Need to set both in one `set_type_parameters` call to avoid
  # conflicts in type parameter constraints of certain wrapper types.
  return set_type_parameters(type, (eltype, parenttype), (param, new_parenttype))
end

@traitfn function set_eltype(
  type::Type{ArrayType}, param
) where {ArrayType <: AbstractArray; !IsWrappedArray{ArrayType}}
  return set_type_parameter(type, eltype, param)
end

# These are generic fallback definitions. By convention,
# this is very commonly true of `AbstractArray` subtypes
# but it may not be correct, but it is very convenient
# to define this to make more operations "just work"
# on most AbstractArrays.
position(type::Type{<:AbstractArray}, ::typeof(eltype)) = Position(1)
position(type::Type{<:AbstractArray}, ::typeof(ndims)) = Position(2)

default_type_parameters(::Type{<:AbstractArray}) = (Float64, 1)

for wrapper in [:PermutedDimsArray, :(Base.ReshapedArray), :SubArray]
  @eval begin
    position(type::Type{<:$wrapper}, ::typeof(eltype)) = Position(1)
    position(type::Type{<:$wrapper}, ::typeof(ndims)) = Position(2)
  end
end
for wrapper in [:(Base.ReshapedArray), :SubArray]
  @eval position(type::Type{<:$wrapper}, ::typeof(parenttype)) = Position(3)
end
for wrapper in [:PermutedDimsArray]
  @eval position(type::Type{<:$wrapper}, ::typeof(parenttype)) = Position(5)
end
