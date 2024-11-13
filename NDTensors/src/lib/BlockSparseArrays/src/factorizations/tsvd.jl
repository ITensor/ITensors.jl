# Truncation schemes
# ------------------
"""
  TuncationScheme

Abstract supertype for all truncated factorization schemes.
See also [`notrunc`](@ref), [`truncdim`](@ref) and [`truncbelow`](@ref).
"""
abstract type TruncationScheme end

"""
  NoTruncation <: TruncationScheme
  notrunc()

Truncation algorithm that represents no truncation. See also [`notrunc`](@ref) for easily
constructing instances of this type.
"""
struct NoTruncation <: TruncationScheme end
notrunc() = NoTruncation()

"""
  TruncAmount <: TruncationScheme
  truncdim(num::Int; by=identity, lt=isless, rev=true)

Truncation algorithm that truncates by keeping the first `num` singular values, sorted using
the kwargs `by`, `lt` and `rev` as passed to the `Base.sort` algorithm.
"""
struct TruncAmount{B,L} <: TruncationScheme
  num::Int
  by::B
  lt::L
  rev::Bool
end
truncdim(n::Int; by=identity, lt=isless, rev=true) = TruncAmount(n, by, lt, rev)

"""
  TruncFilter <: TruncationScheme
  truncbelow(ϵ::Real)

Truncation algorithm that truncates by filter, where `truncbelow` filters all values below a threshold `ϵ`.
"""
struct TruncFilter{F} <: TruncationScheme
  f::F
end
function truncbelow(ϵ::Real)
  @assert ϵ ≥ zero(ϵ)
  return TruncFilter(≥(ϵ))
end

"""
  truncate(F::Factorization; trunc::TruncationScheme) -> Factorization

Truncate a factorization using the given truncation algorithm:
- `trunc = notrunc()` (default): Does nothing.
- `trunc = truncdim(n)`: Keeps the largest `n` values.
- `trunc = truncbelow(ϵ)`: Truncates all values below a threshold `ϵ`.
"""
truncate(F::SVD; trunc::TruncationScheme=notrunc()) = _truncate(F, trunc)

# use _truncate to dispatch on `trunc`
_truncate(usv, ::NoTruncation) = usv

# note: kept implementations separate for possible future ambiguity reasons
function _truncate(usv::SVD, trunc::TruncAmount)
  keep = select_values(usv.S, trunc)
  return SVD(usv.U[:, keep], usv.S[keep], usv.Vt[keep, :])
end
function _truncate(usv::SVD, trunc::TruncFilter)
  keep = select_values(usv.S, trunc)
  return SVD(usv.U[:, keep], usv.S[keep], usv.Vt[keep, :])
end

function select_values(S, trunc::TruncAmount)
  return partialsortperm(S, 1:(trunc.num); trunc.lt, trunc.by, trunc.rev)
end
select_values(S, trunc::TruncFilter) = findall(trunc.f, S)

# For convenience, also add a method to both truncate and decompose
"""
  tsvd(A; full::Bool=false, alg=default_svd_alg(A), trunc=notrunc())

Compute the truncated singular value decomposition (SVD) of `A`.
This is typically achieved by first computing the full SVD, followed by a filtering based on
the computed singular values.
"""
function tsvd(A; kwargs...)
  return tsvd!(eigencopy_oftype(A, LinearAlgebra.eigtype(eltype(A))); kwargs...)
end

"""
  tsvd!(A; full::Bool=false, alg=default_svd_alg(A), trunc=notrunc())

Compute the truncated singular value decomposition (SVD) of `A`, saving space by
overwriting `A` in the process. See documentation of [`tsvd`](@ref) for details.
"""
function tsvd!(A; full::Bool=false, alg=default_svd_alg(A), trunc=notrunc())
  return _tsvd!(A, alg, trunc, full)
end

# default implementation simply dispatches through to `svd` and `truncate`.
function _tsvd!(A, alg, trunc, full)
  F = svd!(A; alg, full)
  return truncate(F; trunc)
end