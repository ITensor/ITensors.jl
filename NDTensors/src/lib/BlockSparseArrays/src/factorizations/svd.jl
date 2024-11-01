using LinearAlgebra:
  LinearAlgebra, Factorization, Algorithm, default_svd_alg, Adjoint, Transpose
using LinearAlgebra: eigencopy_oftype
using BlockArrays: AbstractBlockMatrix, BlockedArray, BlockedMatrix, BlockedVector
using BlockArrays: BlockLayout

# Singular Value Decomposition:
# need new type to deal with U and V having possible different types
# this is basically a carbon copy of the LinearAlgebra implementation.
# additionally, by default we implement a fallback to the LinearAlgebra implementation
# in hope to support as many foreign types as possible that chose to extend those methods.

# TODO: add this to MatrixFactorizations
# TODO: decide where this goes
# TODO: decide whether or not to restrict types to be blocked.
"""
    SVD <: Factorization

Matrix factorization type of the singular value decomposition (SVD) of a matrix `A`.
This is the return type of [`svd(_)`](@ref), the corresponding matrix factorization function.

If `F::SVD` is the factorization object, `U`, `S`, `V` and `Vt` can be obtained
via `F.U`, `F.S`, `F.V` and `F.Vt`, such that `A = U * Diagonal(S) * Vt`.
The singular values in `S` are sorted in descending order.

Iterating the decomposition produces the components `U`, `S`, and `V`.

# Examples
```jldoctest
julia> A = [1. 0. 0. 0. 2.; 0. 0. 3. 0. 0.; 0. 0. 0. 0. 0.; 0. 2. 0. 0. 0.]
4×5 Matrix{Float64}:
 1.0  0.0  0.0  0.0  2.0
 0.0  0.0  3.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  2.0  0.0  0.0  0.0

julia> F = svd(A)
SVD{Float64, Float64, Matrix{Float64}, Vector{Float64}}
U factor:
4×4 Matrix{Float64}:
 0.0  1.0   0.0  0.0
 1.0  0.0   0.0  0.0
 0.0  0.0   0.0  1.0
 0.0  0.0  -1.0  0.0
singular values:
4-element Vector{Float64}:
 3.0
 2.23606797749979
 2.0
 0.0
Vt factor:
4×5 Matrix{Float64}:
 -0.0        0.0  1.0  -0.0  0.0
  0.447214   0.0  0.0   0.0  0.894427
  0.0       -1.0  0.0   0.0  0.0
  0.0        0.0  0.0   1.0  0.0

julia> F.U * Diagonal(F.S) * F.Vt
4×5 Matrix{Float64}:
 1.0  0.0  0.0  0.0  2.0
 0.0  0.0  3.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  2.0  0.0  0.0  0.0

julia> u, s, v = F; # destructuring via iteration

julia> u == F.U && s == F.S && v == F.V
true
```
"""
struct SVD{T,Tr,M<:AbstractArray{T},C<:AbstractVector{Tr},N<:AbstractArray{T}} <:
       Factorization{T}
  U::M
  S::C
  Vt::N
  function SVD{T,Tr,M,C,N}(
    U, S, Vt
  ) where {T,Tr,M<:AbstractArray{T},C<:AbstractVector{Tr},N<:AbstractArray{T}}
    Base.require_one_based_indexing(U, S, Vt)
    return new{T,Tr,M,C,N}(U, S, Vt)
  end
end
function SVD(U::AbstractArray{T}, S::AbstractVector{Tr}, Vt::AbstractArray{T}) where {T,Tr}
  return SVD{T,Tr,typeof(U),typeof(S),typeof(Vt)}(U, S, Vt)
end
function SVD{T}(U::AbstractArray, S::AbstractVector{Tr}, Vt::AbstractArray) where {T,Tr}
  return SVD(
    convert(AbstractArray{T}, U),
    convert(AbstractVector{Tr}, S),
    convert(AbstractArray{T}, Vt),
  )
end

function SVD{T}(F::SVD) where {T}
  return SVD(
    convert(AbstractMatrix{T}, F.U),
    convert(AbstractVector{real(T)}, F.S),
    convert(AbstractMatrix{T}, F.Vt),
  )
end
LinearAlgebra.Factorization{T}(F::SVD) where {T} = SVD{T}(F)

# iteration for destructuring into components
Base.iterate(S::SVD) = (S.U, Val(:S))
Base.iterate(S::SVD, ::Val{:S}) = (S.S, Val(:V))
Base.iterate(S::SVD, ::Val{:V}) = (S.V, Val(:done))
Base.iterate(::SVD, ::Val{:done}) = nothing

function Base.getproperty(F::SVD, d::Symbol)
  if d === :V
    return getfield(F, :Vt)'
  else
    return getfield(F, d)
  end
end

function Base.propertynames(F::SVD, private::Bool=false)
  return private ? (:V, fieldnames(typeof(F))...) : (:U, :S, :V, :Vt)
end

Base.size(A::SVD, dim::Integer) = dim == 1 ? size(A.U, dim) : size(A.Vt, dim)
Base.size(A::SVD) = (size(A, 1), size(A, 2))

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::SVD)
  summary(io, F)
  println(io)
  println(io, "U factor:")
  show(io, mime, F.U)
  println(io, "\nsingular values:")
  show(io, mime, F.S)
  println(io, "\nVt factor:")
  return show(io, mime, F.Vt)
end

Base.adjoint(usv::SVD) = SVD(adjoint(usv.Vt), usv.S, adjoint(usv.U))
Base.transpose(usv::SVD) = SVD(transpose(usv.Vt), usv.S, transpose(usv.U))

# Conversion
Base.AbstractMatrix(F::SVD) = (F.U * Diagonal(F.S)) * F.Vt
Base.AbstractArray(F::SVD) = AbstractMatrix(F)
Base.Matrix(F::SVD) = Array(AbstractArray(F))
Base.Array(F::SVD) = Matrix(F)
SVD(usv::SVD) = usv
SVD(usv::LinearAlgebra.SVD) = SVD(usv.U, usv.S, usv.Vt)

# functions default to LinearAlgebra
# ----------------------------------
"""
    svd!(A; full::Bool = false, alg::Algorithm = default_svd_alg(A)) -> SVD

`svd!` is the same as [`svd`](@ref), but saves space by
overwriting the input `A`, instead of creating a copy. See documentation of [`svd`](@ref) for details.
"""
svd!(A; kwargs...) = SVD(LinearAlgebra.svd!(A; kwargs...))

"""
    svd(A; full::Bool = false, alg::Algorithm = default_svd_alg(A)) -> SVD

Compute the singular value decomposition (SVD) of `A` and return an `SVD` object.

`U`, `S`, `V` and `Vt` can be obtained from the factorization `F` with `F.U`,
`F.S`, `F.V` and `F.Vt`, such that `A = U * Diagonal(S) * Vt`.
The algorithm produces `Vt` and hence `Vt` is more efficient to extract than `V`.
The singular values in `S` are sorted in descending order.

Iterating the decomposition produces the components `U`, `S`, and `V`.

If `full = false` (default), a "thin" SVD is returned. For an ``M
\\times N`` matrix `A`, in the full factorization `U` is ``M \\times M``
and `V` is ``N \\times N``, while in the thin factorization `U` is ``M
\\times K`` and `V` is ``N \\times K``, where ``K = \\min(M,N)`` is the
number of singular values.

`alg` specifies which algorithm and LAPACK method to use for SVD:
- `alg = DivideAndConquer()` (default): Calls `LAPACK.gesdd!`.
- `alg = QRIteration()`: Calls `LAPACK.gesvd!` (typically slower but more accurate) .

!!! compat "Julia 1.3"
    The `alg` keyword argument requires Julia 1.3 or later.

# Examples
```jldoctest
julia> A = rand(4,3);

julia> F = svd(A); # Store the Factorization Object

julia> A ≈ F.U * Diagonal(F.S) * F.Vt
true

julia> U, S, V = F; # destructuring via iteration

julia> A ≈ U * Diagonal(S) * V'
true

julia> Uonly, = svd(A); # Store U only

julia> Uonly == U
true
```
"""
svd(A; kwargs...) =
  SVD(svd!(eigencopy_oftype(A, LinearAlgebra.eigtype(eltype(A))); kwargs...))

LinearAlgebra.svdvals(usv::SVD{<:Any,T}) where {T} = (usv.S)::Vector{T}

