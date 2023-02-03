## Custom `NDTensors.similar` implementation.
## More extensive than `Base.similar`.

# Trait indicating if the AbstractArray type is an array wrapper.
# Assumes that it implements `NDTensors.parenttype`.
@traitdef IsWrappedArray{T}

#! format: off
@traitimpl IsWrappedArray{T} <- is_wrapped_array(T)
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

@traitfn function leaf_parenttype(arraytype::Type{T}) where {T; IsWrappedArray{T}}
  return leaf_parenttype(parenttype(arraytype))
end

@traitfn function leaf_parenttype(arraytype::Type{T}) where {T; !IsWrappedArray{T}}
  return arraytype
end

# For working with instances.
leaf_parenttype(array::AbstractArray) = leaf_parenttype(typeof(array))

# NDTensors.similar
function similar(array::AbstractArray, eltype::Type, dims)
  return NDTensors.similar(similartype(array, eltype, dims), dims)
end
# NDTensors.similar
similar(array::AbstractArray, eltype::Type) = NDTensors.similar(array, eltype, size(array))
# NDTensors.similar
similar(array::AbstractArray, dims) = NDTensors.similar(array, eltype(array), dims)
# NDTensors.similar
similar(array::AbstractArray) = NDTensors.similar(array, eltype(array), size(array))

# NDTensors.similar
# This function actually allocates the data.
function similar(arraytype::Type{<:AbstractArray}, eltype::Type, dims)
  return similartype(arraytype, eltype, dims)(undef, dims)
end
# NDTensors.similar
function similar(arraytype::Type{<:AbstractArray}, eltype::Type)
  return error("Must specify dimensions.")
end
# NDTensors.similar
similar(arraytype::Type{<:AbstractArray}, dims) = similar(arraytype, eltype(arraytype), dims)

function similartype(array::AbstractArray, eltype::Type, dims)
  return similartype(typeof(array), eltype, dims)
end
similartype(array::AbstractArray, eltype::Type) = similartype(array, eltype, size(array))
similartype(array::AbstractArray, dims) = similartype(array, eltype(array), dims)

function similartype(arraytype::Type{<:AbstractArray}, eltype::Type, dims)
  return similartype(similartype(arraytype, eltype), dims)
end

@traitfn function similartype(
  arraytype::Type{T}, eltype::Type
) where {T; !IsWrappedArray{T}}
  return error(
    "The function `similartype(::$T, eltype::Type)` has not been implement for this data type. It is a required part of the NDTensors interface.",
  )
end

@traitfn function similartype(arraytype::Type{T}, dims) where {T; !IsWrappedArray{T}}
  return error(
    "The function `similartype(::$T, dims)` has not been implement for this data type. It is a required part of the NDTensors interface.",
  )
end

## Wrapped arrays
@traitfn function similartype(arraytype::Type{T}, eltype::Type) where {T; IsWrappedArray{T}}
  return similartype(parenttype(arraytype), eltype)
end

@traitfn function similartype(arraytype::Type{T}, dims) where {T; IsWrappedArray{T}}
  return similartype(parenttype(arraytype), eltype)
end

## Overloads needed for `Array`
function similartype(arraytype::Type{<:Array}, eltype::Type)
  return Array{eltype,ndims(arraytype)}
end

function similartype(arraytype::Type{<:Array}, dims)
  return Array{eltype(arraytype),length(dims)}
end
