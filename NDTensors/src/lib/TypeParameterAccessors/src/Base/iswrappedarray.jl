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

struct Self 
end

position(::Type{<:AbstractArray}, ::typeof(parenttype)) = Self()
parameter(type::Type, ::Self) = type

is_wrapped_array(arraytype::Type{<:AbstractArray}) = parenttype(arraytype) ≠ arraytype

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
  @eval position(type::Type{<:$wrapper}, ::typeof(parenttype)) = Position(2)
end
for wrapper in (:ReshapedArray, :SubArray, :StridedView)
  @eval position(type::Type{<:$wrapper}, ::typeof(parenttype)) = Position(3)
end

# For working with instances, not used by
# `SimpleTraits.jl` traits dispatch.
# function is_wrapped_array(arraytype::Type{<:AbstractArray})
#   return parenttype(arraytype) == arraytype
# end

# For working with instances, not used by
# `SimpleTraits.jl` traits dispatch.
is_wrapped_array(array::AbstractArray) = is_wrapped_array(typeof(array))

function set_parenttype(t::Type, parent_type)
  return set_parameter(t, parenttype, parent_type)
end

@traitfn function set_eltype(
  type::Type{ArrayT}, elt::Type
) where {ArrayT <: AbstractArray; IsWrappedArray{ArrayT}}
  new_parenttype = set_eltype(parenttype(type), elt)
  return set_parenttype(set_parameter(type, eltype, elt), new_parenttype)
end
