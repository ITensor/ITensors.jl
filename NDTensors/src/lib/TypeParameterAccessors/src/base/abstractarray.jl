struct Self end

parameter(type::Type, pos::Self) = type
function set_parameter(type::Type, pos::Self, param)
  return error("Can't set the parent type of an unwrapped array type.")
end

function set_eltype(type::Type{<:AbstractArray}, param)
  return error("Not implemented")
end
function set_eltype(array::AbstractArray, param)
  return convert(set_eltype(typeof(array), param), array)
end

function set_ndims(type::Type{<:AbstractArray}, param)
  return error("Not implemented")
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
  return set_parameter(t, parenttype, param)
end

@traitfn function set_eltype(
  type::Type{ArrayType}, param
) where {ArrayType <: AbstractArray; IsWrappedArray{ArrayType}}
  new_parenttype = set_eltype(parenttype(type), param)
  ## return set_parenttype(set_parameter(type, eltype, param), new_parenttype)
  return set_parameter(set_parenttype(type, new_parenttype), eltype, param)
end

@traitfn function set_eltype(
  type::Type{ArrayType}, param
) where {ArrayType <: AbstractArray; !IsWrappedArray{ArrayType}}
  return set_parameter(type, eltype, param)
end

for wrapper in [:(Base.ReshapedArray), :SubArray]
  @eval position(type::Type{<:$wrapper}, ::typeof(parenttype)) = Position(3)
end
for wrapper in [:PermutedDimsArray]
  @eval position(type::Type{<:$wrapper}, ::typeof(parenttype)) = Position(5)
end
