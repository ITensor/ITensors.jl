using ..SparseArrayInterface: Zero
# TODO: Put into `DiagonalArraysSparseArrayDOKsExt`?
using ..SparseArrayDOKs: SparseArrayDOKs, SparseArrayDOK

struct DiagonalArray{T,N,Diag<:AbstractVector{T},Zero} <: AbstractDiagonalArray{T,N}
  diag::Diag
  dims::NTuple{N,Int}
  zero::Zero
end

function DiagonalArray{T,N}(
  diag::AbstractVector{T}, d::Tuple{Vararg{Int,N}}, zero=Zero()
) where {T,N}
  return DiagonalArray{T,N,typeof(diag),typeof(zero)}(diag, d, zero)
end

function DiagonalArray{T,N}(
  diag::AbstractVector, d::Tuple{Vararg{Int,N}}, zero=Zero()
) where {T,N}
  return DiagonalArray{T,N}(T.(diag), d, zero)
end

function DiagonalArray{T,N}(diag::AbstractVector, d::Vararg{Int,N}) where {T,N}
  return DiagonalArray{T,N}(diag, d)
end

function DiagonalArray{T}(
  diag::AbstractVector, d::Tuple{Vararg{Int,N}}, zero=Zero()
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

# Infer size from diagonal
function DiagonalArray{T,N}(diag::AbstractVector) where {T,N}
  return DiagonalArray{T,N}(diag, default_size(diag, N))
end

function DiagonalArray{<:Any,N}(diag::AbstractVector{T}) where {T,N}
  return DiagonalArray{T,N}(diag)
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

# Minimal `AbstractArray` interface
Base.size(a::DiagonalArray) = a.dims

function Base.similar(a::DiagonalArray, elt::Type, dims::Tuple{Vararg{Int}})
  return DiagonalArray{elt}(undef, dims)
end

# Minimal `SparseArrayInterface` interface
SparseArrayInterface.sparse_storage(a::DiagonalArray) = a.diag

# `SparseArrayInterface`
# Defines similar when the output can't be `DiagonalArray`,
# such as in `reshape`.
# TODO: Put into `DiagonalArraysSparseArrayDOKsExt`?
# TODO: Special case 2D to output `SparseMatrixCSC`?
function SparseArrayInterface.sparse_similar(
  a::DiagonalArray, elt::Type, dims::Tuple{Vararg{Int}}
)
  return SparseArrayDOK{elt}(undef, dims)
end

function SparseArrayDOKs.getindex_zero_function(a::DiagonalArray)
  return a.zero
end
