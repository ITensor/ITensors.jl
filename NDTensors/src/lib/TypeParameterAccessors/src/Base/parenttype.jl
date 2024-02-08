using LinearAlgebra: Adjoint, Diagonal, Hermitian, LowerTriangular, Symmetric, Transpose, UnitLowerTriangular, UnitUpperTriangular, UpperTriangular
using Base: ReshapedArray, SubArray
using StridedViews: StridedView
# By default, the `parentype` of an array type is itself
parenttype(arraytype::Type{<:AbstractArray}) = arraytype

# TODO: Use `TypeParameterAccessors` here.
parenttype(t::Type) = get_parameter(t, parenttype_position(t))

for wrap in (:Transpose, :Adjoint, :Symmetric, :Hermitian, :UpperTriangular, :LowerTriangular, :UnitUpperTriangular, :UnitLowerTriangular, :Diagonal)
  @eval parenttype_position(type::$wrap) = 2
end
for wrap in (:ReshapedArray, :SubArray, :StridedView)
  @eval parenttype_position(type::$wrap) = 3
end

"""
Optional definitions for types which are considered `Wrappers` and have a `parenttype`

  Should return a `Position`.
"""
function parenttype_position(type::Type)
  return error(
    "Unable to find the parenttype position of type '$(type)' as it has not been defined."
  )
end

# For working with instances, not used by
# `SimpleTraits.jl` traits dispatch.
parenttype(array::AbstractArray) = parenttype(typeof(array))