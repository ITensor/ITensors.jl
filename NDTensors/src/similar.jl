
#
# Custom NDTensors.similar implementation
# More extensive than Base.similar
#

# Trait indicating if the type is an array wrapper
# Assumes that is implements `Base.parent`.

@traitdef IsWrappedArray{T}

#! format: off
@traitimpl IsWrappedArray{T} <- is_wrapped_array(T)
#! format: on

is_wrapped_array(::Type) = false
is_wrapped_array(::Type{<:ReshapedArray}) = true
is_wrapped_array(::Type{<:Transpose}) = true
is_wrapped_array(::Type{<:Adjoint}) = true
is_wrapped_array(::Type{<:Symmetric}) = true
is_wrapped_array(::Type{<:Hermitian}) = true
is_wrapped_array(::Type{<:UpperTriangular}) = true
is_wrapped_array(::Type{<:LowerTriangular}) = true
is_wrapped_array(::Type{<:UnitUpperTriangular}) = true
is_wrapped_array(::Type{<:UnitLowerTriangular}) = true
is_wrapped_array(::Type{<:Diagonal}) = true
is_wrapped_array(::Type{<:SubArray}) = true

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

# In general define NDTensors.similar = Base.similar
similar(a::Array, args...) = Base.similar(a, args...)
@traitfn similar(a::T, args...) where {T;IsWrappedArray{T}} = Base.similar(a, args...)

similar(::Type{<:Array{T}}, dims) where {T} = Array{T,length(dims)}(undef, dims)

function similar(arraytype::Type{<:AbstractArray}, eltype::Type, size)
  return similar(similartype(arraytype, eltype), size)
end

function similar(arraytype::Type{<:AbstractArray}, size)
  return similar(arraytype, eltype(arraytype), size)
end

#
# similartype returns the type of the object that would be returned by `similar`
#

# TODO: extend to AbstractVector, returning the same type as `Base.similar` would
# (make sure it handles views, CuArrays, etc. correctly)
# This is to help with code that is generic to different storage types.
similartype(arraytype::Type{<:Array}, eltype::Type) = Array{eltype,ndims(arraytype)}

@traitfn function similartype(arraytype::Type{T}, eltype::Type, dims::Tuple) where {T;IsWrappedArray{T}}
  return similartype(parenttype(arraytype), eltype, dims)
end

@traitfn function similartype(arraytype::Type{T}, eltype::Type) where {T;IsWrappedArray{T}}
  return similartype(parenttype(arraytype), eltype)
end

@traitfn function similartype(arraytype::Type{T}) where {T;IsWrappedArray{T}}
  return similartype(parenttype(arraytype))
end
