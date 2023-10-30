module DiagonalArrays

using Compat # allequal
using LinearAlgebra

export DiagonalArray, DiagonalMatrix, DiagonalVector

include("diagview.jl")

struct DefaultZero end

function (::DefaultZero)(eltype::Type, I::CartesianIndex)
  return zero(eltype)
end

struct DiagonalArray{T,N,Diag<:AbstractVector{T},Zero} <: AbstractArray{T,N}
  diag::Diag
  dims::NTuple{N,Int}
  zero::Zero
end

const DiagonalVector{T,Diag,Zero} = DiagonalArray{T,1,Diag,Zero}
const DiagonalMatrix{T,Diag,Zero} = DiagonalArray{T,2,Diag,Zero}

function DiagonalArray{T,N}(
  diag::AbstractVector{T}, d::Tuple{Vararg{Int,N}}, zero=DefaultZero()
) where {T,N}
  return DiagonalArray{T,N,typeof(diag),typeof(zero)}(diag, d, zero)
end

function DiagonalArray{T,N}(
  diag::AbstractVector, d::Tuple{Vararg{Int,N}}, zero=DefaultZero()
) where {T,N}
  return DiagonalArray{T,N}(T.(diag), d, zero)
end

function DiagonalArray{T,N}(diag::AbstractVector, d::Vararg{Int,N}) where {T,N}
  return DiagonalArray{T,N}(diag, d)
end

function DiagonalArray{T}(
  diag::AbstractVector, d::Tuple{Vararg{Int,N}}, zero=DefaultZero()
) where {T,N}
  return DiagonalArray{T,N}(diag, d, zero)
end

function DiagonalArray{T}(diag::AbstractVector, d::Vararg{Int,N}) where {T,N}
  return DiagonalArray{T,N}(diag, d)
end

function DiagonalArray(diag::AbstractVector{T}, d::Tuple{Vararg{Int,N}}) where {T,N}
  return DiagonalArray{T,N}(diag, d)
end

function DiagonalArray(diag::AbstractVector{T}, d::Vararg{Int,N}) where {T,N}
  return DiagonalArray{T,N}(diag, d)
end

default_size(diag::AbstractVector, n) = ntuple(Returns(length(diag)), n)

# Infer size from diagonal
function DiagonalArray{T,N}(diag::AbstractVector) where {T,N}
  return DiagonalArray{T,N}(diag, default_size(diag, N))
end

function DiagonalArray{<:Any,N}(diag::AbstractVector{T}) where {T,N}
  return DiagonalArray{T,N}(diag)
end

function DiagonalMatrix(diag::AbstractVector)
  return DiagonalArray{<:Any,2}(diag)
end

function DiagonalVector(diag::AbstractVector)
  return DiagonalArray{<:Any,1}(diag)
end

# undef
function DiagonalArray{T,N}(::UndefInitializer, d::Tuple{Vararg{Int,N}}) where {T,N}
  return DiagonalArray{T,N}(Vector{T}(undef, minimum(d)), d)
end

function DiagonalArray{T,N}(::UndefInitializer, d::Vararg{Int,N}) where {T,N}
  return DiagonalArray{T,N}(undef, d)
end

function DiagonalArray{T}(::UndefInitializer, d::Tuple{Vararg{Int,N}}) where {T,N}
  return DiagonalArray{T,N}(undef, d)
end

function DiagonalArray{T}(::UndefInitializer, d::Vararg{Int,N}) where {T,N}
  return DiagonalArray{T,N}(undef, d)
end

Base.size(a::DiagonalArray) = a.dims

diagview(a::DiagonalArray) = a.diag
LinearAlgebra.diag(a::DiagonalArray) = copy(diagview(a))

function Base.getindex(a::DiagonalArray{T,N}, I::CartesianIndex{N}) where {T,N}
  i = diagindex(a, I)
  isnothing(i) && return a.zero(T, I)
  return getdiagindex(a, i)
end

function Base.getindex(a::DiagonalArray{T,N}, I::Vararg{Int,N}) where {T,N}
  return getindex(a, CartesianIndex(I))
end

function Base.setindex!(a::DiagonalArray{T,N}, v, I::CartesianIndex{N}) where {T,N}
  i = diagindex(a, I)
  isnothing(i) && return error("Can't set off-diagonal element of DiagonalArray")
  setdiagindex!(a, v, i)
  return a
end

function Base.setindex!(a::DiagonalArray{T,N}, v, I::Vararg{Int,N}) where {T,N}
  a[CartesianIndex(I)] = v
  return a
end

# Make dense.
function densearray(a::DiagonalArray)
  # TODO: Check this works on GPU.
  # TODO: Make use of `a.zero`?
  d = similar(diagview(a), size(a))
  fill!(d, zero(eltype(a)))
  diagcopyto!(d, a)
  return d
end

end
