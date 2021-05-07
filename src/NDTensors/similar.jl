
#
# Custom NDTensors.similar implementation
# More extensive than Base.similar
#

# A union type of AbstractArrays that are wrappers that define `Base.parent`.
const WrappedArray{T,AW} = Union{
  ReshapedArray{T,<:Any,AW},
  Transpose{T,AW},
  Adjoint{T,AW},
  Symmetric{T,AW},
  Hermitian{T,AW},
  UpperTriangular{T,AW},
  LowerTriangular{T,AW},
  UnitUpperTriangular{T,AW},
  UnitLowerTriangular{T,AW},
  Diagonal{T,AW},
  SubArray{T,<:Any,AW},
}

# In general define NDTensors.similar = Base.similar
similar(a::Array, args...) = Base.similar(a, args...)
similar(a::WrappedArray, args...) = Base.similar(a, args...)

# XXX: this is type piracy but why doesn't base have something like this?
# This type piracy is pretty bad, consider making an internal `NDTensors.similar` function.
# Currently Base gives:
# julia> similar(Matrix{Float64}, 2)
# ERROR: MethodError: no method matching Matrix{Float64}(::UndefInitializer, ::Tuple{Int64})
# [...]
similar(::Type{<:Array{T}}, dims) where {T} = Array{T,length(dims)}(undef, dims)

function similar(::Type{ArrayT}, ::Type{ElT}, size) where {ArrayT<:AbstractArray,ElT}
  return similar(similartype(ArrayT, ElT), size)
end

#
# similartype returns the type of the object that would be returned by `similar`
#

# TODO: extend to AbstractVector, returning the same type as `Base.similar` would
# (make sure it handles views, CuArrays, etc. correctly)
# This is to help with code that is generic to different storage types.
similartype(::Type{<:Array{<:Any,N}}, eltype::Type) where {N} = Array{eltype,N}

#similartype(::Type{LinearAlgebra.Adjoint{Float64, Matrix{Float64}}}, ::Type{Float64})

parenttype(::Type{<:WrappedArray{<:Any,P}}) where {P} = P

function similartype(::Type{ArrayT}, eltype::Type, dims::Tuple) where {ArrayT<:WrappedArray}
  return similartype(parenttype(ArrayT), eltype, dims)
end

function similartype(::Type{ArrayT}, eltype::Type) where {ArrayT<:WrappedArray}
  return similartype(parenttype(ArrayT), eltype)
end

function similartype(::Type{ArrayT}) where {ArrayT<:WrappedArray}
  return similartype(parenttype(ArrayT))
end
