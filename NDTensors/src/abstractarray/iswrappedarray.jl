# Trait indicating if the AbstractArray type is an array wrapper.
# Assumes that it implements `NDTensors.parenttype`.
@traitdef IsWrappedArray{ArrayT}

#! format: off
@traitimpl IsWrappedArray{ArrayT} <- is_wrapped_array(ArrayT)
#! format: on

is_wrapped_array(arraytype::Type{<:AbstractArray}) = (parenttype(arraytype) ≠ arraytype)

# TODO: This is only defined because the current design
# of `Diag` using a `Number` as the data type if it
# is a uniform diagonal type. Delete this when it is
# replaced by `DiagonalArray`.
is_wrapped_array(arraytype::Type{<:Number}) = false

# For working with instances, not used by
# `SimpleTraits.jl` traits dispatch.
is_wrapped_array(array::AbstractArray) = is_wrapped_array(typeof(array))

# By default, the `parentype` of an array type is itself
parenttype(arraytype::Type{<:AbstractArray}) = arraytype

# TODO: Use `SetParameters` here.
parenttype(::Type{<:ReshapedArray{<:Any,<:Any,P}}) where {P} = P
parenttype(::Type{<:Transpose{<:Any,P}}) where {P} = P
parenttype(::Type{<:Adjoint{<:Any,P}}) where {P} = P
parenttype(::Type{<:Symmetric{<:Any,P}}) where {P} = P
parenttype(::Type{<:Hermitian{<:Any,P}}) where {P} = P
parenttype(::Type{<:UpperTriangular{<:Any,P}}) where {P} = P
parenttype(::Type{<:LowerTriangular{<:Any,P}}) where {P} = P
parenttype(::Type{<:UnitUpperTriangular{<:Any,P}}) where {P} = P
parenttype(::Type{<:UnitLowerTriangular{<:Any,P}}) where {P} = P
parenttype(::Type{<:Diagonal{<:Any,P}}) where {P} = P
parenttype(::Type{<:SubArray{<:Any,<:Any,P}}) where {P} = P

# For working with instances, not used by
# `SimpleTraits.jl` traits dispatch.
parenttype(array::AbstractArray) = parenttype(typeof(array))

@traitfn function leaf_parenttype(
  arraytype::Type{ArrayT}
) where {ArrayT; IsWrappedArray{ArrayT}}
  return leaf_parenttype(parenttype(arraytype))
end

@traitfn function leaf_parenttype(
  arraytype::Type{ArrayT}
) where {ArrayT; !IsWrappedArray{ArrayT}}
  return arraytype
end

# For working with instances.
leaf_parenttype(array::AbstractArray) = leaf_parenttype(typeof(array))
