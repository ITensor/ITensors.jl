using SimpleTraits: SimpleTraits, @traitdef, @traitimpl
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