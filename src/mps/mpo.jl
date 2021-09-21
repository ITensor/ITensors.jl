
"""
    MPO

A finite size matrix product operator type. 
Keeps track of the orthogonality center.
"""
mutable struct MPO <: AbstractMPS
  data::Vector{ITensor}
  llim::Int
  rlim::Int
end

function MPO(A::Vector{<:ITensor}; ortho_lims::UnitRange=1:length(A))
  return MPO(A, first(ortho_lims) - 1, last(ortho_lims) + 1)
end

MPO() = MPO(ITensor[], 0, 0)

function convert(::Type{MPS}, M::MPO)
  return MPS(data(M); ortho_lims=ortho_lims(M))
end

function convert(::Type{MPO}, M::MPS)
  return MPO(data(M); ortho_lims=ortho_lims(M))
end

function MPO(::Type{ElT}, sites::Vector{<:Index}) where {ElT<:Number}
  N = length(sites)
  v = Vector{ITensor}(undef, N)
  if N == 0
    return MPO()
  elseif N == 1
    v[1] = emptyITensor(ElT, dag(sites[1]), sites[1]')
    return MPO(v)
  end
  space_ii = all(hasqns, sites) ? [QN() => 1] : 1
  l = [Index(space_ii, "Link,l=$ii") for ii in 1:(N - 1)]
  for ii in eachindex(sites)
    s = sites[ii]
    if ii == 1
      v[ii] = emptyITensor(ElT, dag(s), s', l[ii])
    elseif ii == N
      v[ii] = emptyITensor(ElT, dag(l[ii - 1]), dag(s), s')
    else
      v[ii] = emptyITensor(ElT, dag(l[ii - 1]), dag(s), s', l[ii])
    end
  end
  return MPO(v)
end

MPO(sites::Vector{<:Index}) = MPO(Float64, sites)

"""
    MPO(N::Int)

Make an MPO of length `N` filled with default ITensors.
"""
MPO(N::Int) = MPO(Vector{ITensor}(undef, N))

"""
    MPO([::Type{ElT} = Float64}, ]sites, ops::Vector{String})

Make an MPO with pairs of sites `s[i]` and `s[i]'`
and operators `ops` on each site.
"""
function MPO(::Type{ElT}, sites::Vector{<:Index}, ops::Vector{String}) where {ElT<:Number}
  N = length(sites)
  ampo = OpSum() + [ops[n] => n for n in 1:N]
  M = MPO(ampo, sites)

  # Currently, OpSum does not output the optimally truncated
  # MPO (see https://github.com/ITensor/ITensors.jl/issues/526)
  # So here, we need to first normalize, then truncate, then
  # return the normalization.
  lognormM = lognorm(M)
  M ./= exp(lognormM / N)
  truncate!(M; cutoff=1e-15)
  M .*= exp(lognormM / N)
  return M
end

function MPO(::Type{ElT}, sites::Vector{<:Index}, fops::Function) where {ElT<:Number}
  ops = [fops(n) for n in 1:length(sites)]
  return MPO(ElT, sites, ops)
end

MPO(sites::Vector{<:Index}, ops) = MPO(Float64, sites, ops)

"""
    MPO([::Type{ElT} = Float64, ]sites, op::String)

Make an MPO with pairs of sites `s[i]` and `s[i]'`
and operator `op` on every site.
"""
function MPO(::Type{ElT}, sites::Vector{<:Index}, op::String) where {ElT<:Number}
  return MPO(ElT, sites, fill(op, length(sites)))
end

MPO(sites::Vector{<:Index}, op::String) = MPO(Float64, sites, op)

function randomMPO(sites::Vector{<:Index}, m::Int=1)
  M = MPO(sites, "Id")
  for i in eachindex(sites)
    randn!(M[i])
    normalize!(M[i])
  end
  m > 1 && throw(ArgumentError("randomMPO: currently only m==1 supported"))
  return M
end

function MPO(A::ITensor, sites::Vector{<:Index}; kwargs...)
  return MPO(A, IndexSet.(prime.(sites), dag.(sites)); kwargs...)
end

"""
    outer(x::MPS, y::MPS; <keyword argument>) -> MPO

Compute the outer product of `MPS` `x` and `MPS` `y`,
returning an `MPO` approximation.

Note that `y` will be conjugated, and the site indices
of `x` will be primed.

In Dirac notation, this is the operation `|x⟩⟨y|`.

The keyword arguments determine the truncation, and accept
the same arguments as `contract(::MPO, ::MPO; kw...)`.

See also [`product`](@ref), [`contract`](@ref).
"""
function outer(ψ::MPS, ϕ::MPS; kw...)
  ψmat = convert(MPO, ψ')
  ϕmat = convert(MPO, dag(ϕ))
  return contract(ψmat, ϕmat; kw...)
end

"""
    projector(x::MPS; <keyword argument>) -> MPO

Computes the projector onto the state `x`. In Dirac notation, this is the operation `|x⟩⟨x|/|⟨x|x⟩|²`.

Use keyword arguments to control the level of truncation, which are
the same as those accepted by `contract(::MPO, ::MPO; kw...)`.

# Keywords
- `normalize::Bool=true`: whether or not to normalize the input MPS before forming the projector. If `normalize==false` and the input MPS is not already normalized, this function will not output a proper project, and simply outputs `outer(x, x) = |x⟩⟨x|`, i.e. the projector scaled by `norm(x)^2`.
- truncation keyword arguments accepted by `contract(::MPO, ::MPO; kw...)`.

See also [`outer`](@ref), [`contract`](@ref).
"""
function projector(ψ::MPS; normalize::Bool=true, kw...)
  ψψᴴ = outer(ψ, ψ; kw...)
  if normalize
    normalize!(ψψᴴ[orthocenter(ψψᴴ)])
  end
  return ψψᴴ
end

# XXX: rename originalsiteind?
"""
    siteind(M::MPO, j::Int; plev = 0, kwargs...)

Get the first site Index of the MPO found, by
default with prime level 0. 
"""
siteind(M::MPO, j::Int; kwargs...) = siteind(first, M, j; plev=0, kwargs...)

# TODO: make this return the site indices that would have
# been used to create the MPO? I.e.:
# [dag(siteinds(M, j; plev = 0, kwargs...)) for j in 1:length(M)]
"""
    siteinds(M::MPO; kwargs...)

Get a Vector of IndexSets of all the site indices of M.
"""
siteinds(M::MPO; kwargs...) = siteinds(all, M; kwargs...)

function siteinds(Mψ::Tuple{MPO,MPS}, n::Int; kwargs...)
  return siteinds(uniqueinds, Mψ[1], Mψ[2], n; kwargs...)
end

function nsites(Mψ::Tuple{MPO,MPS})
  M, ψ = Mψ
  N = length(M)
  @assert N == length(ψ)
  return N
end

siteinds(Mψ::Tuple{MPO,MPS}; kwargs...) = [siteinds(Mψ, n; kwargs...) for n in 1:nsites(Mψ)]

# XXX: rename originalsiteinds?
"""
    firstsiteinds(M::MPO; kwargs...)

Get a Vector of the first site Index found on each site of M.

By default, it finds the first site Index with prime level 0.
"""
firstsiteinds(M::MPO; kwargs...) = siteinds(first, M; plev=0, kwargs...)

function hassameinds(::typeof(siteinds), ψ::MPS, Hϕ::Tuple{MPO,MPS})
  N = length(ψ)
  @assert N == length(Hϕ[1]) == length(Hϕ[1])
  for n in 1:N
    !hassameinds(siteinds(Hϕ, n), siteinds(ψ, n)) && return false
  end
  return true
end

"""
    dot(y::MPS, A::MPO, x::MPS; make_inds_match::Bool = true)
    inner(y::MPS, A::MPO, x::MPS; make_inds_match::Bool = true)

Compute `<y|A|x> = <y|Ax>.

If `make_inds_match = true`, the function attempts to make
the site indices of `A*x` match with the site indices of `y`
before contracting (so for example, the inputs `y` and `A*x` 
can have different site indices, as long as they have the same 
dimensions or QN blocks).

`A` and `x` must have common site indices.
"""
function dot(y::MPS, A::MPO, x::MPS; make_inds_match::Bool=true)::Number
  N = length(A)
  check_hascommoninds(siteinds, A, x)
  ydag = dag(y)
  sim!(linkinds, ydag)
  if !hassameinds(siteinds, y, (A, x))
    sAx = siteinds((A, x))
    if any(s -> length(s) > 1, sAx)
      n = findfirst(n -> !hassameinds(siteinds(y, n), siteinds((A, x), n)), 1:N)
      error(
        """Calling `dot(ϕ::MPS, H::MPO, ψ::MPS)` with multiple site indices per MPO/MPS tensor but the site indices don't match. Even with `make_inds_match = true`, the case of multiple site indices per MPO/MPS is not handled automatically. The sites with unmatched site indices are:

            inds(ϕ[$n]) = $(inds(y[n]))

            inds(H[$n]) = $(inds(A[n]))

            inds(ψ[$n]) = $(inds(x[n]))

        Make sure the site indices of your MPO/MPS match. You may need to prime one of the MPS, such as `dot(ϕ', H, ψ)`.""",
      )
    end
    if make_inds_match
      replace_siteinds!(ydag, sAx)
    end
  end
  check_hascommoninds(siteinds, A, y)
  O = ydag[1] * A[1] * x[1]
  for j in 2:N
    O = O * ydag[j] * A[j] * x[j]
  end
  return O[]
end

inner(y::MPS, A::MPO, x::MPS; kwargs...) = dot(y, A, x; kwargs...)

"""
    dot(B::MPO, y::MPS, A::MPO, x::MPS; make_inds_match::Bool = true)
    inner(B::MPO, y::MPS, A::MPO, x::MPS; make_inds_match::Bool = true)

Compute `<By|A|x> = <By|Ax>`.

If `make_inds_match = true`, the function attempts to make
the site indices of `A*x` match with the site indices of `B*y`
before contracting (so for example, the inputs `B*y` and `A*x`
can have different site indices, as long as they have the same
dimensions or QN blocks).

`A` and `x` must have common site indices, and `B` and `y`
must have common site indices.
"""
function dot(B::MPO, y::MPS, A::MPO, x::MPS; make_inds_match::Bool=true)::Number
  !make_inds_match && error(
    "make_inds_match = false not currently supported in dot(::MPO, ::MPS, ::MPO, ::MPS)"
  )
  N = length(B)
  if length(y) != N || length(x) != N || length(A) != N
    throw(
      DimensionMismatch(
        "inner: mismatched lengths $N and $(length(x)) or $(length(y)) or $(length(A))"
      ),
    )
  end
  ydag = dag(y)
  prime!(ydag, 2)
  Bdag = dag(B)
  prime!(Bdag)
  # Swap prime levels 1 -> 2 and 2 -> 1.
  for j in eachindex(Bdag)
    Axcommon = commonind(A[j], x[j])
    ABcommon = uniqueind(filterinds(A[j]; tags="Site"), IndexSet(Axcommon))
    swapprime!(Bdag[j], 2, 3)
    swapprime!(Bdag[j], 1, 2)
    swapprime!(Bdag[j], 3, 1)
    noprime!(Bdag[j], prime(ABcommon, 2))
  end
  yB = ydag[1] * Bdag[1]
  Ax = A[1] * x[1]
  O = yB * Ax
  for j in 2:N
    yB = ydag[j] * Bdag[j]
    Ax = A[j] * x[j]
    yB *= O
    O = yB * Ax
  end
  return O[]
end

# TODO: maybe make these into tuple inputs?
# Also can generalize to:
# inner((β, B, y), (α, A, x))
inner(B::MPO, y::MPS, A::MPO, x::MPS) = dot(B, y, A, x)

function dot(M1::MPO, M2::MPO; make_inds_match::Bool=false)
  if make_inds_match
    error("In dot(::MPO, ::MPO), make_inds_match is not currently supported")
  end
  return _log_or_not_dot(M1, M2, false; make_inds_match=make_inds_match)
end

# TODO: implement by combing the MPO indices and converting
# to MPS
function logdot(M1::MPO, M2::MPO; make_inds_match::Bool=false)
  if make_inds_match
    error("In dot(::MPO, ::MPO), make_inds_match is not currently supported")
  end
  return _log_or_not_dot(M1, M2, true; make_inds_match=make_inds_match)
end

function tr(M::MPO; plev::Pair{Int,Int}=0 => 1, tags::Pair=ts"" => ts"")
  N = length(M)
  #
  # TODO: choose whether to contract or trace
  # first depending on the bond dimension. The scaling is:
  #
  # 1. Trace last:  O(χ²d²) + O(χd²)
  # 2. Trace first: O(χ²d²) + O(χ²)
  #
  # So tracing first is better if d > √χ.
  #
  L = tr(M[1]; plev=plev, tags=tags)
  for j in 2:N
    L *= M[j]
    L = tr(L; plev=plev, tags=tags)
  end
  return L
end

"""
    error_contract(y::MPS, A::MPO, x::MPS;
                   make_inds_match::Bool = true)
    error_contract(y::MPS, x::MPS, x::MPO;
                   make_inds_match::Bool = true)

Compute the distance between A|x> and an approximation MPS y:
`| |y> - A|x> |/| A|x> | = √(1 + (<y|y> - 2*real(<y|A|x>))/<Ax|A|x>)`.

If `make_inds_match = true`, the function attempts match the site 
indices of `y` with the site indices of `A` that are not common
with `x`.
"""
function error_contract(y::MPS, A::MPO, x::MPS; kwargs...)
  N = length(A)
  if length(y) != N || length(x) != N
    throw(
      DimensionMismatch("inner: mismatched lengths $N and $(length(x)) or $(length(y))")
    )
  end
  iyy = dot(y, y; kwargs...)
  iyax = dot(y, A, x; kwargs...)
  iaxax = dot(A, x, A, x; kwargs...)
  return sqrt(abs(1.0 + (iyy - 2 * real(iyax)) / iaxax))
end

error_contract(y::MPS, x::MPS, A::MPO) = error_contract(y, A, x)

function contract(A::MPO, ψ::MPS; kwargs...)
  method = get(kwargs, :method, "densitymatrix")

  # Keyword argument deprecations
  if method == "DensityMatrix"
    @warn "In contract, method DensityMatrix is deprecated in favor of densitymatrix"
    method = "densitymatrix"
  end

  if method == "Naive"
    @warn "In contract, method Naive is deprecated in favor of naive"
    method = "naive"
  end

  if method == "densitymatrix"
    Aψ = _contract_densitymatrix(A, ψ; kwargs...)
  elseif method == "naive"
    Aψ = _contract_naive(A, ψ; kwargs...)
  else
    throw(ArgumentError("Method $method not supported"))
  end
  return Aψ
end

contract_mpo_mps_doc = """
    contract(ψ::MPS, A::MPO; kwargs...) -> MPS
    *(::MPS, ::MPO; kwargs...) -> MPS

    contract(A::MPO, ψ::MPS; kwargs...) -> MPS
    *(::MPO, ::MPS; kwargs...) -> MPS

Contract the `MPO` `A` with the `MPS` `ψ`, returning an `MPS` with the unique
site indices of the `MPO`.

Choose the method with the `method` keyword, for example
`"densitymatrix"` and `"naive"`.

# Keywords
- `cutoff::Float64=1e-13`: the cutoff value for truncating the density matrix eigenvalues. Note that the default is somewhat arbitrary and subject to change, in general you should set a `cutoff` value.
- `maxdim::Int=maxlinkdim(A) * maxlinkdim(ψ))`: the maximal bond dimension of the results MPS.
- `mindim::Int=1`: the minimal bond dimension of the resulting MPS.
- `normalize::Bool=false`: whether or not to normalize the resulting MPS.
- `method::String="densitymatrix"`: the algorithm to use for the contraction.
"""

@doc """
$contract_mpo_mps_doc
""" contract(::MPO, ::MPS)

contract(ψ::MPS, A::MPO; kwargs...) = contract(A, ψ; kwargs...)

*(A::MPO, B::MPS; kwargs...) = contract(A, B; kwargs...)
*(A::MPS, B::MPO; kwargs...) = contract(A, B; kwargs...)

# TODO: try this to copy the docstring
# Causing an error in Revise
#@doc """
#$contract_mpo_mps_doc
#""" *(::MPO, ::MPS)

#@doc (@doc contract(::MPO, ::MPS)) *(::MPO, ::MPS)

function _contract_densitymatrix(A::MPO, ψ::MPS; kwargs...)::MPS
  n = length(A)
  n != length(ψ) &&
    throw(DimensionMismatch("lengths of MPO ($n) and MPS ($(length(ψ))) do not match"))
  if n == 1
    return MPS([A[1] * ψ[1]])
  end

  ψ_out = similar(ψ)
  cutoff::Float64 = get(kwargs, :cutoff, 1e-13)
  requested_maxdim::Int = get(kwargs, :maxdim, maxlinkdim(A) * maxlinkdim(ψ))
  mindim::Int = max(get(kwargs, :mindim, 1), 1)
  normalize::Bool = get(kwargs, :normalize, false)

  any(i -> isempty(i), siteinds(commoninds, A, ψ)) &&
    error("In `contract(A::MPO, x::MPS)`, `A` and `x` must share a set of site indices")

  # In case A and ψ have the same link indices
  A = sim(linkinds, A)

  ψ_c = dag(ψ)
  A_c = dag(A)

  # To not clash with the link indices of A and ψ
  sim!(linkinds, A_c)
  sim!(linkinds, ψ_c)
  sim!(siteinds, commoninds, A_c, ψ_c)

  # A version helpful for making the density matrix
  simA_c = sim(siteinds, uniqueinds, A_c, ψ_c)

  # Store the left environment tensors
  E = Vector{ITensor}(undef, n - 1)

  E[1] = ψ[1] * A[1] * A_c[1] * ψ_c[1]
  for j in 2:(n - 1)
    E[j] = E[j - 1] * ψ[j] * A[j] * A_c[j] * ψ_c[j]
  end
  R = ψ[n] * A[n]
  simR_c = ψ_c[n] * simA_c[n]
  ρ = E[n - 1] * R * simR_c
  l = linkind(ψ, n - 1)
  ts = isnothing(l) ? "" : tags(l)
  Lis = siteinds(uniqueinds, A, ψ, n)
  Ris = siteinds(uniqueinds, simA_c, ψ_c, n)
  F = eigen(ρ, Lis, Ris; ishermitian=true, tags=ts, kwargs...)
  D, U, Ut = F.D, F.V, F.Vt
  l_renorm, r_renorm = F.l, F.r
  ψ_out[n] = Ut
  R = R * dag(Ut) * ψ[n - 1] * A[n - 1]
  simR_c = simR_c * U * ψ_c[n - 1] * simA_c[n - 1]
  for j in reverse(2:(n - 1))
    # Determine smallest maxdim to use
    cip = commoninds(ψ[j], E[j - 1])
    ciA = commoninds(A[j], E[j - 1])
    prod_dims = dim(cip) * dim(ciA)
    maxdim = min(prod_dims, requested_maxdim)

    s = siteinds(uniqueinds, A, ψ, j)
    s̃ = siteinds(uniqueinds, simA_c, ψ_c, j)
    ρ = E[j - 1] * R * simR_c
    l = linkind(ψ, j - 1)
    ts = isnothing(l) ? "" : tags(l)
    Lis = IndexSet(s..., l_renorm)
    Ris = IndexSet(s̃..., r_renorm)
    F = eigen(ρ, Lis, Ris; ishermitian=true, maxdim=maxdim, tags=ts, kwargs...)
    D, U, Ut = F.D, F.V, F.Vt
    l_renorm, r_renorm = F.l, F.r
    ψ_out[j] = Ut
    R = R * dag(Ut) * ψ[j - 1] * A[j - 1]
    simR_c = simR_c * U * ψ_c[j - 1] * simA_c[j - 1]
  end
  if normalize
    R ./= norm(R)
  end
  ψ_out[1] = R
  setleftlim!(ψ_out, 0)
  setrightlim!(ψ_out, 2)
  return ψ_out
end

function _contract_naive(A::MPO, ψ::MPS; kwargs...)::MPS
  truncate = get(kwargs, :truncate, true)

  N = length(A)
  if N != length(ψ)
    throw(DimensionMismatch("lengths of MPO ($N) and MPS ($(length(ψ))) do not match"))
  end

  ψ_out = MPS(N)
  for j in 1:N
    ψ_out[j] = A[j] * ψ[j]
  end

  for b in 1:(N - 1)
    Al = commonind(A[b], A[b + 1])
    pl = commonind(ψ[b], ψ[b + 1])
    C = combiner(Al, pl)
    ψ_out[b] *= C
    ψ_out[b + 1] *= dag(C)
  end

  if truncate
    truncate!(ψ_out; kwargs...)
  end

  return ψ_out
end

function contract(A::MPO, B::MPO; kwargs...)
  cutoff::Float64 = get(kwargs, :cutoff, 1e-14)
  resp_degen::Bool = get(kwargs, :respect_degenerate, true)
  maxdim::Int = get(kwargs, :maxdim, maxlinkdim(A) * maxlinkdim(B))
  mindim::Int = max(get(kwargs, :mindim, 1), 1)
  N = length(A)
  N != length(B) &&
    throw(DimensionMismatch("lengths of MPOs A ($N) and B ($(length(B))) do not match"))
  # Special case for a single site
  N == 1 && return MPO([A[1] * B[1]])
  A = orthogonalize(A, 1)
  B = orthogonalize(B, 1)
  A = sim(linkinds, A)
  sA = siteinds(uniqueinds, A, B)
  sB = siteinds(uniqueinds, B, A)
  C = MPO(N)
  lCᵢ = Index[]
  R = ITensor(1)
  for i in 1:(N - 2)
    RABᵢ = R * A[i] * B[i]
    left_inds = [sA[i]..., sB[i]..., lCᵢ...]
    C[i], R = factorize(
      RABᵢ,
      left_inds;
      ortho="left",
      tags=commontags(linkinds(A, i)),
      cutoff=cutoff,
      maxdim=maxdim,
      mindim=mindim,
      kwargs...,
    )
    lCᵢ = dag(commoninds(C[i], R))
  end
  i = N - 1
  RABᵢ = R * A[i] * B[i] * A[i + 1] * B[i + 1]
  left_inds = [sA[i]..., sB[i]..., lCᵢ...]
  C[N - 1], C[N] = factorize(
    RABᵢ,
    left_inds;
    ortho="right",
    tags=commontags(linkinds(A, i)),
    cutoff=cutoff,
    maxdim=maxdim,
    mindim=mindim,
    kwargs...,
  )
  truncate!(C; kwargs...)
  return C
end

contract_mpo_mpo_doc = """
    contract(A::MPO, B::MPO; kwargs...) -> MPO
    *(::MPO, ::MPO; kwargs...) -> MPO

Contract the `MPO` `A` with the `MPO` `B`, returning an `MPO` with the 
site indices that are not shared between `A` and `B`.

# Keywords
- `cutoff::Float64=1e-13`: the cutoff value for truncating the density matrix eigenvalues. Note that the default is somewhat arbitrary and subject to change, in general you should set a `cutoff` value.
- `maxdim::Int=maxlinkdim(A) * maxlinkdim(B))`: the maximal bond dimension of the results MPS.
- `mindim::Int=1`: the minimal bond dimension of the resulting MPS.
"""

@doc """
$contract_mpo_mpo_doc
""" contract(::MPO, ::MPO)

*(A::MPO, B::MPO; kwargs...) = contract(A, B; kwargs...)

# TODO: try this to copy the docstring
# Causing an error in Revise
#@doc """
#$contract_mpo_mpo_doc
#""" *(::MPO, ::MPO)

#@doc (@doc contract(::MPO, ::MPO)) *(::MPO, ::MPO)

"""
    sample(M::MPO)

Given a normalized MPO `M`,
returns a `Vector{Int}` of `length(M)`
corresponding to one sample of the
probability distribution defined by the MPO,
treating the MPO as a density matrix.

The MPO `M` should have an (approximately)
positive spectrum.
"""
function sample(M::MPO)
  N = length(M)
  s = siteinds(M)
  R = Vector{ITensor}(undef, N)
  R[N] = M[N] * δ(dag(s[N]))
  for n in reverse(1:(N - 1))
    R[n] = M[n] * δ(dag(s[n])) * R[n + 1]
  end

  if abs(1.0 - R[1][]) > 1E-8
    error("sample: MPO is not normalized, norm=$(norm(M[1]))")
  end

  result = zeros(Int, N)
  ρj = M[1] * R[2]
  Lj = ITensor()

  for j in 1:N
    s = siteind(M, j)
    d = dim(s)
    # Compute the probability of each state
    # one-by-one and stop when the random
    # number r is below the total prob so far
    pdisc = 0.0
    r = rand()
    # Will need n, An, and pn below
    n = 1
    projn = ITensor()
    pn = 0.0
    while n <= d
      projn = ITensor(s)
      projn[s[n]] = 1.0
      pnc = (ρj * projn * prime(projn))[]
      if imag(pnc) > 1e-8
        error("In sample, probability $pnc is complex.")
      end
      pn = real(pnc)
      pdisc += pn
      (r < pdisc) && break
      n += 1
    end
    result[j] = n
    if j < N
      if j == 1
        Lj = M[j] * projn * prime(projn)
      elseif j > 1
        Lj = Lj * M[j] * projn * prime(projn)
      end
      if j == N - 1
        ρj = Lj * M[j + 1]
      else
        ρj = Lj * M[j + 1] * R[j + 2]
      end
      s = siteind(M, j + 1)
      normj = (ρj * δ(s', s))[]
      ρj ./= normj
    end
  end
  return result
end

function HDF5.write(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, M::MPO)
  g = create_group(parent, name)
  attributes(g)["type"] = "MPO"
  attributes(g)["version"] = 1
  N = length(M)
  write(g, "rlim", M.rlim)
  write(g, "llim", M.llim)
  write(g, "length", N)
  for n in 1:N
    write(g, "MPO[$(n)]", M[n])
  end
end

function HDF5.read(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{MPO})
  g = open_group(parent, name)
  if read(attributes(g)["type"]) != "MPO"
    error("HDF5 group or file does not contain MPO data")
  end
  N = read(g, "length")
  rlim = read(g, "rlim")
  llim = read(g, "llim")
  v = [read(g, "MPO[$(i)]", ITensor) for i in 1:N]
  return MPO(v, llim, rlim)
end
