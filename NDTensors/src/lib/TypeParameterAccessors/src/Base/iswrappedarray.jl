using SimpleTraits
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

parenttype(::Type) = nothing

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
  @eval position(type::Type{<:$wrapper}, ::typeof(parenttype)) = 2
end
for wrapper in (:ReshapedArray, :SubArray, :StridedView)
  @eval position(type::Type{<:$wrapper},::typeof(parenttype)) = 3
end

# For working with instances, not used by
# `SimpleTraits.jl` traits dispatch.
function is_wrapped_array(arraytype::Type{<:AbstractArray})
  return (position(arraytype, parenttype) ≠ UndefinedPosition())
end

# For working with instances, not used by
# `SimpleTraits.jl` traits dispatch.
is_wrapped_array(array::AbstractArray) = is_wrapped_array(typeof(array))

function set_parenttype(t::Type, parent_type)
  return set_parameter(t, parenttype, parent_type)
end

# By default, the `parentype` of an array type is itself
## TODO when both of these functions are defined as `parenttype`
## a warning is thrown by the compiler
@traitfn parenttype(
  arraytype::Type{ArrayT}
) where {ArrayT <: AbstractArray; !IsWrappedArray{ArrayT}} = arraytype

## TODO I am not sure why this is throwing a Warning
@traitfn parenttype(
  wrapper::Type{WrapperT}
) where {WrapperT <: AbstractArray; IsWrappedArray{WrapperT}} =
  parameter(wrapper, position(wrapper, parenttype))

@traitfn function set_eltype(
  type::Type{ArrayT}, elt::Type
) where {ArrayT <: AbstractArray; IsWrappedArray{ArrayT}}
  new_parenttype = set_eltype(parenttype(type), elt)
  return set_parenttype(set_parameter(type, eltype, elt), new_parenttype)
end
