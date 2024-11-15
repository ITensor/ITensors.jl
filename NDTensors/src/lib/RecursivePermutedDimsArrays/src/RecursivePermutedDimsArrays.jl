# Mostly copied from https://github.com/JuliaLang/julia/blob/master/base/permuteddimsarray.jl
# Like `PermutedDimsArrays` but recursive, similar to `Adjoint` and `Transpose`.
module RecursivePermutedDimsArrays

import Base: permutedims, permutedims!
export RecursivePermutedDimsArray

# Some day we will want storage-order-aware iteration, so put perm in the parameters
struct RecursivePermutedDimsArray{T,N,perm,iperm,AA<:AbstractArray} <: AbstractArray{T,N}
  parent::AA

  function RecursivePermutedDimsArray{T,N,perm,iperm,AA}(
    data::AA
  ) where {T,N,perm,iperm,AA<:AbstractArray}
    (isa(perm, NTuple{N,Int}) && isa(iperm, NTuple{N,Int})) ||
      error("perm and iperm must both be NTuple{$N,Int}")
    isperm(perm) ||
      throw(ArgumentError(string(perm, " is not a valid permutation of dimensions 1:", N)))
    all(d -> iperm[perm[d]] == d, 1:N) ||
      throw(ArgumentError(string(perm, " and ", iperm, " must be inverses")))
    return new(data)
  end
end

"""
    RecursivePermutedDimsArray(A, perm) -> B

Given an AbstractArray `A`, create a view `B` such that the
dimensions appear to be permuted. Similar to `permutedims`, except
that no copying occurs (`B` shares storage with `A`).

See also [`permutedims`](@ref), [`invperm`](@ref).

# Examples
```jldoctest
julia> A = rand(3,5,4);

julia> B = RecursivePermutedDimsArray(A, (3,1,2));

julia> size(B)
(4, 3, 5)

julia> B[3,1,2] == A[1,2,3]
true
```
"""
Base.@constprop :aggressive function RecursivePermutedDimsArray(
  data::AbstractArray{T,N}, perm
) where {T,N}
  length(perm) == N ||
    throw(ArgumentError(string(perm, " is not a valid permutation of dimensions 1:", N)))
  iperm = invperm(perm)
  return RecursivePermutedDimsArray{
    recursivepermuteddimsarraytype(T, perm),N,(perm...,),(iperm...,),typeof(data)
  }(
    data
  )
end

# Ideally we would use `Base.promote_op(recursivepermuteddimsarray, type, perm)` but
# that doesn't seem to preserve the `perm`/`iperm` type parameters.
function recursivepermuteddimsarraytype(type::Type{<:AbstractArray{<:AbstractArray}}, perm)
  return RecursivePermutedDimsArray{
    recursivepermuteddimsarraytype(eltype(type), perm),ndims(type),perm,invperm(perm),type
  }
end
function recursivepermuteddimsarraytype(type::Type{<:AbstractArray}, perm)
  return PermutedDimsArray{eltype(type),ndims(type),perm,invperm(perm),type}
end
recursivepermuteddimsarraytype(type::Type, perm) = type

function recursivepermuteddimsarray(A::AbstractArray{<:AbstractArray}, perm)
  return RecursivePermutedDimsArray(A, perm)
end
recursivepermuteddimsarray(A::AbstractArray, perm) = PermutedDimsArray(A, perm)
# By default, assume scalar and don't permute.
recursivepermuteddimsarray(x, perm) = x

Base.parent(A::RecursivePermutedDimsArray) = A.parent
function Base.size(A::RecursivePermutedDimsArray{T,N,perm}) where {T,N,perm}
  return genperm(size(parent(A)), perm)
end
function Base.axes(A::RecursivePermutedDimsArray{T,N,perm}) where {T,N,perm}
  return genperm(axes(parent(A)), perm)
end
Base.has_offset_axes(A::RecursivePermutedDimsArray) = Base.has_offset_axes(A.parent)
function Base.similar(A::RecursivePermutedDimsArray, T::Type, dims::Base.Dims)
  return similar(parent(A), T, dims)
end
function Base.cconvert(::Type{Ptr{T}}, A::RecursivePermutedDimsArray{T}) where {T}
  return Base.cconvert(Ptr{T}, parent(A))
end

# It's OK to return a pointer to the first element, and indeed quite
# useful for wrapping C routines that require a different storage
# order than used by Julia. But for an array with unconventional
# storage order, a linear offset is ambiguous---is it a memory offset
# or a linear index?
function Base.pointer(A::RecursivePermutedDimsArray, i::Integer)
  throw(
    ArgumentError(
      "pointer(A, i) is deliberately unsupported for RecursivePermutedDimsArray"
    ),
  )
end

function Base.strides(A::RecursivePermutedDimsArray{T,N,perm}) where {T,N,perm}
  s = strides(parent(A))
  return ntuple(d -> s[perm[d]], Val(N))
end
function Base.elsize(
  ::Type{<:RecursivePermutedDimsArray{<:Any,<:Any,<:Any,<:Any,P}}
) where {P}
  return Base.elsize(P)
end

@inline function Base.getindex(
  A::RecursivePermutedDimsArray{T,N,perm,iperm}, I::Vararg{Int,N}
) where {T,N,perm,iperm}
  @boundscheck checkbounds(A, I...)
  @inbounds val = recursivepermuteddimsarray(getindex(A.parent, genperm(I, iperm)...), perm)
  return val
end
@inline function Base.setindex!(
  A::RecursivePermutedDimsArray{T,N,perm,iperm}, val, I::Vararg{Int,N}
) where {T,N,perm,iperm}
  @boundscheck checkbounds(A, I...)
  @inbounds setindex!(A.parent, recursivepermuteddimsarray(val, perm), genperm(I, iperm)...)
  return val
end

function Base.isassigned(
  A::RecursivePermutedDimsArray{T,N,perm,iperm}, I::Vararg{Int,N}
) where {T,N,perm,iperm}
  @boundscheck checkbounds(Bool, A, I...) || return false
  @inbounds x = isassigned(A.parent, genperm(I, iperm)...)
  return x
end

@inline genperm(I::NTuple{N,Any}, perm::Dims{N}) where {N} = ntuple(d -> I[perm[d]], Val(N))
@inline genperm(I, perm::AbstractVector{Int}) = genperm(I, (perm...,))

function Base.copyto!(
  dest::RecursivePermutedDimsArray{T,N}, src::AbstractArray{T,N}
) where {T,N}
  checkbounds(dest, axes(src)...)
  return _copy!(dest, src)
end
Base.copyto!(dest::RecursivePermutedDimsArray, src::AbstractArray) = _copy!(dest, src)

function _copy!(P::RecursivePermutedDimsArray{T,N,perm}, src) where {T,N,perm}
  # If dest/src are "close to dense," then it pays to be cache-friendly.
  # Determine the first permuted dimension
  d = 0  # d+1 will hold the first permuted dimension of src
  while d < ndims(src) && perm[d + 1] == d + 1
    d += 1
  end
  if d == ndims(src)
    copyto!(parent(P), src) # it's not permuted
  else
    R1 = CartesianIndices(axes(src)[1:d])
    d1 = findfirst(isequal(d + 1), perm)::Int  # first permuted dim of dest
    R2 = CartesianIndices(axes(src)[(d + 2):(d1 - 1)])
    R3 = CartesianIndices(axes(src)[(d1 + 1):end])
    _permutedims!(P, src, R1, R2, R3, d + 1, d1)
  end
  return P
end

@noinline function _permutedims!(
  P::RecursivePermutedDimsArray, src, R1::CartesianIndices{0}, R2, R3, ds, dp
)
  ip, is = axes(src, dp), axes(src, ds)
  for jo in first(ip):8:last(ip), io in first(is):8:last(is)
    for I3 in R3, I2 in R2
      for j in jo:min(jo + 7, last(ip))
        for i in io:min(io + 7, last(is))
          @inbounds P[i, I2, j, I3] = src[i, I2, j, I3]
        end
      end
    end
  end
  return P
end

@noinline function _permutedims!(P::RecursivePermutedDimsArray, src, R1, R2, R3, ds, dp)
  ip, is = axes(src, dp), axes(src, ds)
  for jo in first(ip):8:last(ip), io in first(is):8:last(is)
    for I3 in R3, I2 in R2
      for j in jo:min(jo + 7, last(ip))
        for i in io:min(io + 7, last(is))
          for I1 in R1
            @inbounds P[I1, i, I2, j, I3] = src[I1, i, I2, j, I3]
          end
        end
      end
    end
  end
  return P
end

const CommutativeOps = Union{
  typeof(+),
  typeof(Base.add_sum),
  typeof(min),
  typeof(max),
  typeof(Base._extrema_rf),
  typeof(|),
  typeof(&),
}

function Base._mapreduce_dim(
  f,
  op::CommutativeOps,
  init::Base._InitialValue,
  A::RecursivePermutedDimsArray,
  dims::Colon,
)
  return Base._mapreduce_dim(f, op, init, parent(A), dims)
end
function Base._mapreduce_dim(
  f::typeof(identity),
  op::Union{typeof(Base.mul_prod),typeof(*)},
  init::Base._InitialValue,
  A::RecursivePermutedDimsArray{<:Union{Real,Complex}},
  dims::Colon,
)
  return Base._mapreduce_dim(f, op, init, parent(A), dims)
end

function Base.mapreducedim!(
  f,
  op::CommutativeOps,
  B::AbstractArray{T,N},
  A::RecursivePermutedDimsArray{S,N,perm,iperm},
) where {T,S,N,perm,iperm}
  C = RecursivePermutedDimsArray{T,N,iperm,perm,typeof(B)}(B) # make the inverse permutation for the output
  Base.mapreducedim!(f, op, C, parent(A))
  return B
end
function Base.mapreducedim!(
  f::typeof(identity),
  op::Union{typeof(Base.mul_prod),typeof(*)},
  B::AbstractArray{T,N},
  A::RecursivePermutedDimsArray{<:Union{Real,Complex},N,perm,iperm},
) where {T,N,perm,iperm}
  C = RecursivePermutedDimsArray{T,N,iperm,perm,typeof(B)}(B) # make the inverse permutation for the output
  Base.mapreducedim!(f, op, C, parent(A))
  return B
end

function Base.showarg(
  io::IO, A::RecursivePermutedDimsArray{T,N,perm}, toplevel
) where {T,N,perm}
  print(io, "RecursivePermutedDimsArray(")
  Base.showarg(io, parent(A), false)
  print(io, ", ", perm, ')')
  toplevel && print(io, " with eltype ", eltype(A))
  return nothing
end

end