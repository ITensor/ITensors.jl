# Mostly copied from https://github.com/JuliaLang/julia/blob/master/base/permuteddimsarray.jl
# Like `PermutedDimsArrays` but singly nested, similar to `Adjoint` and `Transpose`
# (though those are fully recursive).
#=
TODO: Investigate replacing this with a `PermutedDimsArray` wrapped around a `MappedArrays.MappedArray`.
There are a few issues with that:
1. Just using a type alias leads to type piracy, for example the constructor is type piracy.
2. `setindex!(::NestedPermutedDimsArray, I...)` fails because no conversion is defined between `Array`
and `PermutedDimsArray`.
3. The type alias is tricky to define, ideally it would have similar type parameters to the current
`NestedPermutedDimsArrays.NestedPermutedDimsArray` definition which matches the type parameters
of `PermutedDimsArrays.PermutedDimsArray` but that seems to be difficult to achieve.
```julia
module NestedPermutedDimsArrays

using MappedArrays: MultiMappedArray, mappedarray
export NestedPermutedDimsArray

const NestedPermutedDimsArray{TT,T<:AbstractArray{TT},N,perm,iperm,AA<:AbstractArray{T}} = PermutedDimsArray{
  PermutedDimsArray{TT,N,perm,iperm,T},
  N,
  perm,
  iperm,
  MultiMappedArray{
    PermutedDimsArray{TT,N,perm,iperm,T},
    N,
    Tuple{AA},
    Type{PermutedDimsArray{TT,N,perm,iperm,T}},
    Type{PermutedDimsArray{TT,N,iperm,perm,T}},
  },
}

function NestedPermutedDimsArray(a::AbstractArray, perm)
  iperm = invperm(perm)
  f = PermutedDimsArray{eltype(eltype(a)),ndims(a),perm,iperm,eltype(a)}
  finv = PermutedDimsArray{eltype(eltype(a)),ndims(a),iperm,perm,eltype(a)}
  return PermutedDimsArray(mappedarray(f, finv, a), perm)
end

end
```
=#
module NestedPermutedDimsArrays

import Base: permutedims, permutedims!
export NestedPermutedDimsArray

# Some day we will want storage-order-aware iteration, so put perm in the parameters
struct NestedPermutedDimsArray{T,N,perm,iperm,AA<:AbstractArray} <: AbstractArray{T,N}
  parent::AA

  function NestedPermutedDimsArray{T,N,perm,iperm,AA}(
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
    NestedPermutedDimsArray(A, perm) -> B

Given an AbstractArray `A`, create a view `B` such that the
dimensions appear to be permuted. Similar to `permutedims`, except
that no copying occurs (`B` shares storage with `A`).

See also [`permutedims`](@ref), [`invperm`](@ref).

# Examples
```jldoctest
julia> A = rand(3,5,4);

julia> B = NestedPermutedDimsArray(A, (3,1,2));

julia> size(B)
(4, 3, 5)

julia> B[3,1,2] == A[1,2,3]
true
```
"""
Base.@constprop :aggressive function NestedPermutedDimsArray(
  data::AbstractArray{T,N}, perm
) where {T,N}
  length(perm) == N ||
    throw(ArgumentError(string(perm, " is not a valid permutation of dimensions 1:", N)))
  iperm = invperm(perm)
  return NestedPermutedDimsArray{
    PermutedDimsArray{eltype(T),N,(perm...,),(iperm...,),T},
    N,
    (perm...,),
    (iperm...,),
    typeof(data),
  }(
    data
  )
end

Base.parent(A::NestedPermutedDimsArray) = A.parent
function Base.size(A::NestedPermutedDimsArray{T,N,perm}) where {T,N,perm}
  return genperm(size(parent(A)), perm)
end
function Base.axes(A::NestedPermutedDimsArray{T,N,perm}) where {T,N,perm}
  return genperm(axes(parent(A)), perm)
end
Base.has_offset_axes(A::NestedPermutedDimsArray) = Base.has_offset_axes(A.parent)
function Base.similar(A::NestedPermutedDimsArray, T::Type, dims::Base.Dims)
  return similar(parent(A), T, dims)
end
function Base.cconvert(::Type{Ptr{T}}, A::NestedPermutedDimsArray{T}) where {T}
  return Base.cconvert(Ptr{T}, parent(A))
end

# It's OK to return a pointer to the first element, and indeed quite
# useful for wrapping C routines that require a different storage
# order than used by Julia. But for an array with unconventional
# storage order, a linear offset is ambiguous---is it a memory offset
# or a linear index?
function Base.pointer(A::NestedPermutedDimsArray, i::Integer)
  throw(
    ArgumentError("pointer(A, i) is deliberately unsupported for NestedPermutedDimsArray")
  )
end

function Base.strides(A::NestedPermutedDimsArray{T,N,perm}) where {T,N,perm}
  s = strides(parent(A))
  return ntuple(d -> s[perm[d]], Val(N))
end
function Base.elsize(::Type{<:NestedPermutedDimsArray{<:Any,<:Any,<:Any,<:Any,P}}) where {P}
  return Base.elsize(P)
end

@inline function Base.getindex(
  A::NestedPermutedDimsArray{T,N,perm,iperm}, I::Vararg{Int,N}
) where {T,N,perm,iperm}
  @boundscheck checkbounds(A, I...)
  @inbounds val = PermutedDimsArray(getindex(A.parent, genperm(I, iperm)...), perm)
  return val
end
@inline function Base.setindex!(
  A::NestedPermutedDimsArray{T,N,perm,iperm}, val, I::Vararg{Int,N}
) where {T,N,perm,iperm}
  @boundscheck checkbounds(A, I...)
  @inbounds setindex!(A.parent, PermutedDimsArray(val, iperm), genperm(I, iperm)...)
  return val
end

function Base.isassigned(
  A::NestedPermutedDimsArray{T,N,perm,iperm}, I::Vararg{Int,N}
) where {T,N,perm,iperm}
  @boundscheck checkbounds(Bool, A, I...) || return false
  @inbounds x = isassigned(A.parent, genperm(I, iperm)...)
  return x
end

@inline genperm(I::NTuple{N,Any}, perm::Dims{N}) where {N} = ntuple(d -> I[perm[d]], Val(N))
@inline genperm(I, perm::AbstractVector{Int}) = genperm(I, (perm...,))

function Base.copyto!(
  dest::NestedPermutedDimsArray{T,N}, src::AbstractArray{T,N}
) where {T,N}
  checkbounds(dest, axes(src)...)
  return _copy!(dest, src)
end
Base.copyto!(dest::NestedPermutedDimsArray, src::AbstractArray) = _copy!(dest, src)

function _copy!(P::NestedPermutedDimsArray{T,N,perm}, src) where {T,N,perm}
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
  P::NestedPermutedDimsArray, src, R1::CartesianIndices{0}, R2, R3, ds, dp
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

@noinline function _permutedims!(P::NestedPermutedDimsArray, src, R1, R2, R3, ds, dp)
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
  f, op::CommutativeOps, init::Base._InitialValue, A::NestedPermutedDimsArray, dims::Colon
)
  return Base._mapreduce_dim(f, op, init, parent(A), dims)
end
function Base._mapreduce_dim(
  f::typeof(identity),
  op::Union{typeof(Base.mul_prod),typeof(*)},
  init::Base._InitialValue,
  A::NestedPermutedDimsArray{<:Union{Real,Complex}},
  dims::Colon,
)
  return Base._mapreduce_dim(f, op, init, parent(A), dims)
end

function Base.mapreducedim!(
  f, op::CommutativeOps, B::AbstractArray{T,N}, A::NestedPermutedDimsArray{S,N,perm,iperm}
) where {T,S,N,perm,iperm}
  C = NestedPermutedDimsArray{T,N,iperm,perm,typeof(B)}(B) # make the inverse permutation for the output
  Base.mapreducedim!(f, op, C, parent(A))
  return B
end
function Base.mapreducedim!(
  f::typeof(identity),
  op::Union{typeof(Base.mul_prod),typeof(*)},
  B::AbstractArray{T,N},
  A::NestedPermutedDimsArray{<:Union{Real,Complex},N,perm,iperm},
) where {T,N,perm,iperm}
  C = NestedPermutedDimsArray{T,N,iperm,perm,typeof(B)}(B) # make the inverse permutation for the output
  Base.mapreducedim!(f, op, C, parent(A))
  return B
end

function Base.showarg(
  io::IO, A::NestedPermutedDimsArray{T,N,perm}, toplevel
) where {T,N,perm}
  print(io, "NestedPermutedDimsArray(")
  Base.showarg(io, parent(A), false)
  print(io, ", ", perm, ')')
  toplevel && print(io, " with eltype ", eltype(A))
  return nothing
end

end
