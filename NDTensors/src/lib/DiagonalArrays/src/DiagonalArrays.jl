module DiagonalArrays

using Compat # allequal
using LinearAlgebra

export DiagonalArray, DiagonalMatrix, DiagonalVector, DiagIndex, DiagIndices, densearray

include("diagview.jl")

# TODO: Make use of `Zero` type in `SparseArrayInterface`.
struct DefaultZero end

function (::DefaultZero)(eltype::Type, I::CartesianIndex)
  return zero(eltype)
end

# TODO: Rename `diag` to `nonzero_values`, to make more
# generic?
struct DiagonalArray{T,N,Diag<:AbstractVector{T},Zero} <: AbstractArray{T,N}
  diag::Diag
  dims::NTuple{N,Int}
  zero::Zero
end

# TODO: Use `Accessors.jl`.
# TODO: Make more generic, call `set_nonzero_values`.
set_diag(a::DiagonalArray, diag) = DiagonalArray(diag, size(a), a.zero)

Base.copy(a::DiagonalArray) = set_diag(a, copy(a[DiagIndices()]))
Base.similar(a::DiagonalArray) = set_diag(a, similar(a[DiagIndices()]))

Base.size(a::DiagonalArray) = a.dims

# TODO: Rename `storage_values`.
diagview(a::DiagonalArray) = a.diag
LinearAlgebra.diag(a::DiagonalArray) = copy(diagview(a))

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

# TODO: Use `sparse_getindex` from `SparseArrayInterface`.
function Base.getindex(a::DiagonalArray{T,N}, I::CartesianIndex{N}) where {T,N}
  i = diagindex(a, I)
  isnothing(i) && return a.zero(T, I)
  return a[DiagIndex(i)]
end

function Base.getindex(a::DiagonalArray{T,N}, I::Vararg{Int,N}) where {T,N}
  return a[CartesianIndex(I)]
end

# TODO: Use `sparse_setindex!` from `SparseArrayInterface`.
function Base.setindex!(a::DiagonalArray{T,N}, v, I::CartesianIndex{N}) where {T,N}
  i = diagindex(a, I)
  isnothing(i) && return error("Can't set off-diagonal element of DiagonalArray")
  a[DiagIndex(i)] = v
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
  # TODO: Use `getindex_zero(a)` or `zero_value(a)`.
  fill!(d, zero(eltype(a)))
  diagcopyto!(d, a)
  return d
end

# TODO: Make generic in `SparseArrayInterface`, use a `SparseIndices(:)`
# iterator?
function Base.permutedims!(a_dest::AbstractArray, a_src::DiagonalArray, perm)
  @assert ndims(a_src) == ndims(a_dest) == length(perm)
  a_dest[DiagIndices()] = a_src[DiagIndices()]
  return a_dest
end

# TODO: Should this copy? `LinearAlgebra.Diagonal` does not copy
# with `permutedims`.
# TODO: Use `copy(a)` or `permutedims!!`? May be better for immutable diagonals.
function Base.permutedims(a::DiagonalArray, perm)
  a_dest = similar(a)
  permutedims!(a_dest, a, perm)
  return a_dest
end

# This function is used by Julia broadcasting for sparse arrays
# to decide how to allocated the output:
# LinearAlgebra.fzeropreserving(Broadcast.Broadcasted(f, (a,)))
# If it preserves zeros value, it keeps it structured, if not
# it allocates an Array.
# TODO: Introduce an `IsSparse` trait? This would be generic
# for any type that implements `map_nonzeros!`.
function Base.map(f, a::DiagonalArray)
  # TODO: Test `iszero(f(x))`.
  return map_nonzeros(f, a)
end

# TODO: Make this generic in `SparseArrayInterface`.
function map_diag(f, a::DiagonalArray)
  return set_diag(a, map(f, a[DiagIndices()]))
end

# API from `SparseArray`/`BlockSparseArray`
function map_nonzeros(f, a::DiagonalArray)
  return map_diag(f, a)
end

# TODO: Introduce an `IsSparse` trait? This would be generic
# for any type that implements `map_nonzeros!`.
function Base.map!(f, a_dest::AbstractArray, a_src::DiagonalArray)
  # TODO: Test `iszero(f(x))`.
  map_nonzeros!(f, a_dest, a_src)
  return a_dest
end

function map_diag!(f, a_dest::AbstractArray, a_src::DiagonalArray)
  map!(f, a_dest[DiagIndices()], a_src[DiagIndices()])
  return a_dest
end

# API from `SparseArray`/`BlockSparseArray`
function map_nonzeros!(f, a_dest::AbstractArray, a_src::DiagonalArray)
  map_diag!(f, a_dest, a_src)
  return a_dest
end

const DiagonalMatrix{T,Diag,Zero} = DiagonalArray{T,2,Diag,Zero}

function DiagonalMatrix(diag::AbstractVector)
  return DiagonalArray{<:Any,2}(diag)
end

const DiagonalVector{T,Diag,Zero} = DiagonalArray{T,1,Diag,Zero}

function DiagonalVector(diag::AbstractVector)
  return DiagonalArray{<:Any,1}(diag)
end

end
