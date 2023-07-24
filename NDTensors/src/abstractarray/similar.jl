## Custom `NDTensors.similar` implementation.
## More extensive than `Base.similar`.

# Trait indicating if the AbstractArray type is an array wrapper.
# Assumes that it implements `NDTensors.parenttype`.
@traitdef IsWrappedArray{ArrayT}

#! format: off
@traitimpl IsWrappedArray{ArrayT} <- is_wrapped_array(ArrayT)
#! format: on

is_wrapped_array(arraytype::Type{<:AbstractArray}) = (parenttype(arraytype) ≠ arraytype)

# For working with instances, not used by
# `SimpleTraits.jl` traits dispatch.
is_wrapped_array(array::AbstractArray) = is_wrapped_array(typeof(array))

# By default, the `parentype` of an array type is itself
parenttype(arraytype::Type{<:AbstractArray}) = arraytype

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

# This function actually allocates the data.
# NDTensors.similar
function similar(arraytype::Type{<:AbstractArray}, dims::Tuple)
  shape = NDTensors.to_shape(arraytype, dims)
  return similartype(arraytype, shape)(undef, NDTensors.to_shape(arraytype, shape))
end

# This function actually allocates the data.
# Catches conversions of dimensions specified by ranges
# dimensions specified by integers with `Base.to_shape`.
# NDTensors.similar
function similar(arraytype::Type{<:AbstractArray}, dims::Dims)
  return similartype(arraytype, dims)(undef, dims)
end

# NDTensors.similar
function similar(arraytype::Type{<:AbstractArray}, dims::DimOrInd...)
  return similar(arraytype, NDTensors.to_shape(dims))
end

# Handles range inputs, `Base.to_shape` converts them to integer dimensions.
# See Julia's `base/abstractarray.jl`.
# NDTensors.similar
function similar(
  arraytype::Type{<:AbstractArray},
  shape::Tuple{Union{Integer,OneTo},Vararg{Union{Integer,OneTo}}},
)
  return NDTensors.similar(arraytype, NDTensors.to_shape(shape))
end

# NDTensors.similar
function similar(arraytype::Type{<:AbstractArray}, eltype::Type, dims::Tuple)
  return NDTensors.similar(similartype(arraytype, eltype, dims), dims)
end

# TODO: Add an input `structure` which can store things like the nonzero
# structure of a sparse/block sparse tensor.
# NDTensors.similar
# function similar(arraytype::Type{<:AbstractArray}, structure)
#   return NDTensors.similar(similartype(arraytype, structure), structure)
# end

# TODO: Add an input `structure` which can store things like the nonzero
# structure of a sparse/block sparse tensor.
# NDTensors.similar
# function similar(arraytype::Type{<:AbstractArray}, eltype::Type, structure)
#   return NDTensors.similar(similartype(arraytype, eltype, structure), structure)
# end

# TODO: Add an input `structure` which can store things like the nonzero
# structure of a sparse/block sparse tensor.
# NDTensors.similar
# function similar(arraytype::Type{<:AbstractArray}, structure, dims::Tuple)
#   return NDTensors.similar(similartype(arraytype, structure, dims), structure, dims)
# end

# TODO: Add an input `structure` which can store things like the nonzero
# structure of a sparse/block sparse tensor.
# NDTensors.similar
# function similar(arraytype::Type{<:AbstractArray}, eltype::Type, structure, dims::Tuple)
#   return NDTensors.similar(similartype(arraytype, eltype, structure, dims), structure, dims)
# end

# TODO: Maybe makes an empty array, i.e. `similartype(arraytype, eltype)()`?
# NDTensors.similar
function similar(arraytype::Type{<:AbstractArray}, eltype::Type)
  return error("Must specify dimensions.")
end

## NDTensors.similar for instances

# NDTensors.similar
function similar(array::AbstractArray, eltype::Type, dims::Tuple)
  return NDTensors.similar(similartype(typeof(array), eltype), dims)
end

# NDTensors.similar
similar(array::AbstractArray, dims::Tuple) = NDTensors.similar(typeof(array), dims)

# Use the `size` to determine the dimensions
# NDTensors.similar
function similar(array::AbstractArray, eltype::Type)
  return NDTensors.similar(typeof(array), eltype, size(array))
end

# Use the `size` to determine the dimensions
# NDTensors.similar
similar(array::AbstractArray) = NDTensors.similar(typeof(array), size(array))

## similartype

function similartype(arraytype::Type{<:AbstractArray}, eltype::Type, dims::Tuple)
  return similartype(similartype(arraytype, eltype), dims)
end

@traitfn function similartype(
  arraytype::Type{ArrayT}, eltype::Type
) where {ArrayT; !IsWrappedArray{ArrayT}}
  return set_eltype(arraytype, eltype)
end

@traitfn function similartype(
  arraytype::Type{ArrayT}, dims::Tuple
) where {ArrayT; !IsWrappedArray{ArrayT}}
  return set_indstype(arraytype, dims)
end

function similartype(arraytype::Type{<:AbstractArray}, dims::DimOrInd...)
  return similartype(arraytype, dims)
end

function similartype(arraytype::Type{<:AbstractArray})
  return similartype(arraytype, eltype(arraytype))
end

## Wrapped arrays
@traitfn function similartype(
  arraytype::Type{ArrayT}, eltype::Type
) where {ArrayT; IsWrappedArray{ArrayT}}
  return similartype(parenttype(arraytype), eltype)
end

@traitfn function similartype(
  arraytype::Type{ArrayT}, dims::Tuple
) where {ArrayT; IsWrappedArray{ArrayT}}
  return similartype(parenttype(arraytype), dims)
end

# This is for uniform `Diag` storage which uses
# a Number as the data type.
# TODO: Delete this when we change to using a
# `FillArray` instead. This is a stand-in
# to make things work with the current design.
function similartype(numbertype::Type{<:Number})
  return numbertype
end

# Instances
function similartype(array::AbstractArray, eltype::Type, dims...)
  return similartype(typeof(array), eltype, dims...)
end
similartype(array::AbstractArray, eltype::Type) = similartype(typeof(array), eltype)
similartype(array::AbstractArray, dims...) = similartype(typeof(array), dims...)
