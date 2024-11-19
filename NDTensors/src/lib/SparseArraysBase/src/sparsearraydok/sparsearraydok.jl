using Accessors: @set
using Dictionaries: Dictionary, set!
using MacroTools: @capture

# TODO: Parametrize by `data`?
struct SparseArrayDOK{T,N,Zero} <: AbstractSparseArray{T,N}
  data::Dictionary{CartesianIndex{N},T}
  dims::Ref{NTuple{N,Int}}
  zero::Zero
  function SparseArrayDOK{T,N,Zero}(data, dims::NTuple{N,Int}, zero) where {T,N,Zero}
    return new{T,N,Zero}(data, Ref(dims), zero)
  end
end

# Constructors
function SparseArrayDOK(data, dims::Tuple{Vararg{Int}}, zero)
  return SparseArrayDOK{eltype(data),length(dims),typeof(zero)}(data, dims, zero)
end

function SparseArrayDOK{T,N,Zero}(dims::Tuple{Vararg{Int}}, zero) where {T,N,Zero}
  return SparseArrayDOK{T,N,Zero}(default_data(T, N), dims, zero)
end

function SparseArrayDOK{T,N}(dims::Tuple{Vararg{Int}}, zero) where {T,N}
  return SparseArrayDOK{T,N,typeof(zero)}(dims, zero)
end

function SparseArrayDOK{T,N}(dims::Tuple{Vararg{Int}}) where {T,N}
  return SparseArrayDOK{T,N}(dims, default_zero())
end

function SparseArrayDOK{T}(dims::Tuple{Vararg{Int}}) where {T}
  return SparseArrayDOK{T,length(dims)}(dims)
end

function SparseArrayDOK{T}(dims::Int...) where {T}
  return SparseArrayDOK{T}(dims)
end

# Specify zero function
function SparseArrayDOK{T}(dims::Tuple{Vararg{Int}}, zero) where {T}
  return SparseArrayDOK{T,length(dims)}(dims, zero)
end

# undef
function SparseArrayDOK{T,N,Zero}(
  ::UndefInitializer, dims::Tuple{Vararg{Int}}, zero
) where {T,N,Zero}
  return SparseArrayDOK{T,N,Zero}(dims, zero)
end

function SparseArrayDOK{T,N}(::UndefInitializer, dims::Tuple{Vararg{Int}}, zero) where {T,N}
  return SparseArrayDOK{T,N}(dims, zero)
end

function SparseArrayDOK{T,N}(::UndefInitializer, dims::Tuple{Vararg{Int}}) where {T,N}
  return SparseArrayDOK{T,N}(dims)
end

function SparseArrayDOK{T}(::UndefInitializer, dims::Tuple{Vararg{Int}}) where {T}
  return SparseArrayDOK{T}(dims)
end

# Axes version
function SparseArrayDOK{T}(
  ::UndefInitializer, axes::Tuple{Vararg{AbstractUnitRange}}
) where {T}
  @assert all(isone, first.(axes))
  return SparseArrayDOK{T}(length.(axes))
end

function SparseArrayDOK{T}(::UndefInitializer, dims::Int...) where {T}
  return SparseArrayDOK{T}(dims...)
end

function SparseArrayDOK{T}(::UndefInitializer, dims::Tuple{Vararg{Int}}, zero) where {T}
  return SparseArrayDOK{T}(dims, zero)
end

# Base `AbstractArray` interface
Base.size(a::SparseArrayDOK) = a.dims[]

getindex_zero_function(a::SparseArrayDOK) = a.zero
function set_getindex_zero_function(a::SparseArrayDOK, f)
  return @set a.zero = f
end

function setindex_notstored!(
  a::SparseArrayDOK{<:Any,N}, value, I::CartesianIndex{N}
) where {N}
  set!(sparse_storage(a), I, value)
  return a
end

function Base.similar(a::SparseArrayDOK, elt::Type, dims::Tuple{Vararg{Int}})
  return SparseArrayDOK{elt}(undef, dims, getindex_zero_function(a))
end

# `SparseArraysBase` interface
sparse_storage(a::SparseArrayDOK) = a.data

function dropall!(a::SparseArrayDOK)
  return empty!(sparse_storage(a))
end

SparseArrayDOK(a::AbstractArray) = SparseArrayDOK{eltype(a)}(a)

SparseArrayDOK{T}(a::AbstractArray) where {T} = SparseArrayDOK{T,ndims(a)}(a)

function SparseArrayDOK{T,N}(a::AbstractArray) where {T,N}
  return sparse_convert(SparseArrayDOK{T,N}, a)
end

function Base.resize!(a::SparseArrayDOK{<:Any,N}, new_size::NTuple{N,Integer}) where {N}
  a.dims[] = new_size
  return a
end

function setindex_maybe_grow!(a::SparseArrayDOK{<:Any,N}, value, I::Vararg{Int,N}) where {N}
  if any(I .> size(a))
    resize!(a, max.(I, size(a)))
  end
  a[I...] = value
  return a
end

function is_setindex!_expr(expr::Expr)
  return is_assignment_expr(expr) && is_getindex_expr(first(expr.args))
end
is_setindex!_expr(x) = false

is_getindex_expr(expr::Expr) = (expr.head === :ref)
is_getindex_expr(x) = false

is_assignment_expr(expr::Expr) = (expr.head === :(=))
is_assignment_expr(expr) = false

macro maybe_grow(expr)
  if !is_setindex!_expr(expr)
    error(
      "@maybe_grow must be used with setindex! syntax (as @maybe_grow a[i,j,...] = value)"
    )
  end
  @capture(expr, array_[indices__] = value_)
  return :(setindex_maybe_grow!($(esc(array)), $(esc(value)), $(esc.(indices)...)))
end
