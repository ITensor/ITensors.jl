using SimpleTraits: SimpleTraits, Not, @traitdef, @traitimpl, @traitfn
using LinearAlgebra:
  Adjoint,
  Diagonal,
  Hermitian,
  LowerTriangular,
  Symmetric,
  Transpose,
  UnitLowerTriangular,
  UnitUpperTriangular,
  UpperTriangular
using Base: ReshapedArray, SubArray
using StridedViews: StridedView

# Trait indicating if the AbstractArray type is an array wrapper.
# Assumes that it implements `NDTensors.parenttype`.
@traitdef IsWrappedArray{ArrayT}

#! format: off
@traitimpl IsWrappedArray{ArrayT} <- is_wrapped_array(ArrayT)
#! format: on

parenttype(type::Type{<:AbstractArray}) = parameter(type, parenttype)
parenttype(object::AbstractArray) = parenttype(typeof(object))
position(::Type{<:AbstractArray}, ::typeof(parenttype)) = Self()

is_wrapped_array(arraytype::Type{<:AbstractArray}) = (parenttype(arraytype) ≠ arraytype)
is_wrapped_array(array::AbstractArray) = is_wrapped_array(typeof(array))

for wrapper in (
  :Transpose,
  :Adjoint,
  :Symmetric,
  :Hermitian,
  :UpperTriangular,
  :LowerTriangular,
  :UnitUpperTriangular,
  :UnitLowerTriangular,
  :Diagonal,
)
  @eval position(::Type{<:$wrapper}, ::typeof(parenttype)) = Position(2)
  @eval position_name(::Type{<:$wrapper}, ::Position{2}) = parenttype
end
for wrapper in (:ReshapedArray, :SubArray, :StridedView)
  @eval position(type::Type{<:$wrapper}, ::typeof(parenttype)) = Position(3)
  @eval position_name(::Type{<:$wrapper}, ::Position{3}) = parenttype
end

## These functions will be used in place of unwrap_type but will be
## call indirectly through the expose function.
@traitfn function unwrap_array_type(
  arraytype::Type{ArrayT}
) where {ArrayT; IsWrappedArray{ArrayT}}
  return unwrap_type(parenttype(arraytype))
end

@traitfn function unwrap_array_type(
  arraytype::Type{ArrayT}
) where {ArrayT; !IsWrappedArray{ArrayT}}
  return arraytype
end

# For working with instances.
unwrap_array_type(array::AbstractArray) = unwrap_type(typeof(array))

function set_parenttype(t::Type, parent_type)
  return set_parameter(t, parenttype, parent_type)
end

@traitfn function set_eltype(
  type::Type{ArrayT}, elt::Type
) where {ArrayT <: AbstractArray; IsWrappedArray{ArrayT}}
  new_parenttype = set_eltype(parenttype(type), elt)
  return set_parenttype(set_parameter(type, eltype, elt), new_parenttype)
end

@traitfn function set_eltype(
  type::Type{ArrayT}, elt::Type
) where {ArrayT <: AbstractArray; !IsWrappedArray{ArrayT}}
  return set_parameter(type, eltype, elt)
end
