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

"""
Optional definitions for types which are considered `Wrappers` and have a `parenttype`

  Should return a `Position`.
"""
function parenttype_position(type::Type)
  return 0
end

for wrap in (
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
  @eval parenttype_position(type::Type{<:$wrap}) = 2
end
for wrap in (:ReshapedArray, :SubArray, :StridedView)
  @eval parenttype_position(type::Type{<:$wrap}) = 3
end

# By default, the `parentype` of an array type is itself
parenttype(arraytype::Type{<:AbstractArray}) = arraytype

## TODO I am not sure why this is throwing a Warning
@traitfn parenttype(
  wrapper::Type{ArrayT}
) where {ArrayT <: AbstractArray; IsWrappedArray{ArrayT}} =
  parameter(wrapper, parenttype_position(wrapper))

# For working with instances, not used by
# `SimpleTraits.jl` traits dispatch.
# parenttype(array::AbstractArray) = parenttype(typeof(array))

is_wrapped_array(arraytype::Type{<:AbstractArray}) = (parenttype_position(arraytype) ≠ 0)

# TODO: This is only defined because the current design
# of `Diag` using a `Number` as the data type if it
# is a uniform diagonal type. Delete this when it is
# replaced by `DiagonalArray`.
is_wrapped_array(arraytype::Type{<:Number}) = false

# For working with instances, not used by
# `SimpleTraits.jl` traits dispatch.
is_wrapped_array(array::AbstractArray) = is_wrapped_array(typeof(array))
