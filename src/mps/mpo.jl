
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

set_data(A::MPO, data::Vector{ITensor}) = MPO(data, A.llim, A.rlim)

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
function MPO(::Type{ElT}, sites::Vector{<:Index}, ops::Vector) where {ElT<:Number}
  N = length(sites)
  os = Prod{Op}()
  for n in 1:N
    os *= Op(ops[n], n)
  end
  M = MPO(os, sites)

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

function MPO(sites::Vector{<:Index}, os::OpSum)
  return error(
    "To construct an MPO from an OpSum `opsum` and a set of indices `sites`, you must use MPO(opsum, sites)",
  )
end

"""
    MPO([::Type{ElT} = Float64, ]sites, op::String)

Make an MPO with pairs of sites `s[i]` and `s[i]'`
and operator `op` on every site.
"""
function MPO(::Type{ElT}, sites::Vector{<:Index}, op::String) where {ElT<:Number}
  return MPO(ElT, sites, fill(op, length(sites)))
end

MPO(sites::Vector{<:Index}, op::String) = MPO(Float64, sites, op)

function MPO(::Type{ElT}, sites::Vector{<:Index}, op::Matrix{<:Number}) where {ElT<:Number}
  # return MPO(ElT, sites, fill(op, length(sites)))
  return error(
    "Not defined on purpose because of potential ambiguity with `MPO(A::Array, sites::Vector)`. Pass the on-site matrices as functions like `MPO(sites, n -> [1 0; 0 1])` instead.",
  )
end

MPO(sites::Vector{<:Index}, op::Matrix{ElT}) where {ElT<:Number} = MPO(ElT, sites, op)

function randomMPO(sites::Vector{<:Index}, m::Int=1)
  return randomMPO(Random.default_rng(), sites, m)
end

function randomMPO(rng::AbstractRNG, sites::Vector{<:Index}, m::Int=1)
  M = MPO(sites, "Id")
  for i in eachindex(sites)
    randn!(rng, M[i])
    normalize!(M[i])
  end
  m > 1 && throw(ArgumentError("randomMPO: currently only m==1 supported"))
  return M
end

function MPO(A::ITensor, sites::Vector{<:Index}; kwargs...)
  return MPO(A, IndexSet.(prime.(sites), dag.(sites)); kwargs...)
end

function outer_mps_mps_deprecation_warning()
  return "Calling `outer(ψ::MPS, ϕ::MPS)` for MPS `ψ` and `ϕ` with shared indices is deprecated. Currently, we automatically prime `ψ` to make sure the site indices don't clash, but that will no longer be the case in ITensors v0.4. To upgrade your code, call `outer(ψ', ϕ)`. Although the new interface seems less convenient, it will allow `outer` to accept more general outer products going forward, such as outer products where some indices are shared (a batched outer product) or outer products of MPS between site indices that aren't just related by a single prime level."
end

function deprecate_make_inds_unmatch(::typeof(outer), ψ::MPS, ϕ::MPS; kw...)
  if hassameinds(siteinds, ψ, ϕ)
    warn_once(outer_mps_mps_deprecation_warning(), :outer_mps_mps)
    ψ = ψ'
  end
  return ψ, ϕ
end

"""
    outer(x::MPS, y::MPS; <keyword argument>) -> MPO

Compute the outer product of `MPS` `x` and `MPS` `y`,
returning an `MPO` approximation. Note that `y` will be conjugated.

In Dirac notation, this is the operation `|x⟩⟨y|`.

If you want an outer product of an MPS with itself, you should
call `outer(x', x; kwargs...)` so that the resulting MPO
has site indices with indices coming in pairs of prime levels
of 1 and 0. If not, the site indices won't be unique which would
not be an outer product.

For example:

```julia
s = siteinds("S=1/2", 5)
x = randomMPS(s)
y = randomMPS(s)
outer(x, y) # Incorrect! Site indices must be unique.
outer(x', y) # Results in an MPO with pairs of primed and unprimed indices.
```

This allows for more general outer products, such as more general
MPO outputs which don't have pairs of primed and unprimed indices,
or outer products where the input MPS are vectorizations of MPOs.

For example:

```julia
s = siteinds("S=1/2", 5)
X = MPO(s, "Id")
Y = MPO(s, "Id")
x = convert(MPS, X)
y = convert(MPS, Y)
outer(x, y) # Incorrect! Site indices must be unique.
outer(x', y) # Incorrect! Site indices must be unique.
outer(addtags(x, "Out"), addtags(y, "In")) # This performs a proper outer product.
```

The keyword arguments determine the truncation, and accept
the same arguments as `contract(::MPO, ::MPO; kwargs...)`.

See also [`apply`](@ref), [`contract`](@ref).
"""
function outer(ψ::MPS, ϕ::MPS; kw...)
  ψ, ϕ = deprecate_make_inds_unmatch(outer, ψ, ϕ; kw...)

  ψmat = convert(MPO, ψ)
  ϕmat = convert(MPO, dag(ϕ))
  return contract(ψmat, ϕmat; kw...)
end

"""
    projector(x::MPS; <keyword argument>) -> MPO

Computes the projector onto the state `x`. In Dirac notation, this is the operation `|x⟩⟨x|/|⟨x|x⟩|²`.

Use keyword arguments to control the level of truncation, which are
the same as those accepted by `contract(::MPO, ::MPO; kw...)`.

# Keywords

  - `normalize::Bool=true`: whether or not to normalize the input MPS before
     forming the projector. If `normalize==false` and the input MPS is not
     already normalized, this function will not output a proper project, and
     simply outputs `outer(x, x) = |x⟩⟨x|`, i.e. the projector scaled by `norm(x)^2`.
  - truncation keyword arguments accepted by `contract(::MPO, ::MPO; kw...)`.

See also [`outer`](@ref), [`contract`](@ref).
"""
function projector(ψ::MPS; normalize::Bool=true, kw...)
  ψψᴴ = outer(ψ', ψ; kw...)
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

function inner_mps_mpo_mps_deprecation_warning()
  return """
 Calling `inner(x::MPS, A::MPO, y::MPS)` where the site indices of the `MPS`
 `x` and the `MPS` resulting from contracting `MPO` `A` with `MPS` `y` don't
 match is deprecated as of ITensors v0.3 and will result in an error in ITensors
 v0.4. The most common cause of this is something like the following:

 ```julia
 s = siteinds("S=1/2")
 psi = randomMPS(s)
 H = MPO(s, "Id")
 inner(psi, H, psi)
 ```

 `psi` has the Index structure `-s-(psi)` and `H` has the Index structure
 `-s'-(H)-s-`, so the Index structure of would be `(dag(psi)-s- -s'-(H)-s-(psi)`
  unless the prime levels were fixed. Previously we tried fixing the prime level
   in situations like this, but we will no longer be doing that going forward.

 There are a few ways to fix this. You can simply change:

 ```julia
 inner(psi, H, psi)
 ```

 to:

 ```julia
 inner(psi', H, psi)
 ```

 in which case the Index structure will be `(dag(psi)-s'-(H)-s-(psi)`.

 Alternatively, you can use the `Apply` function:

 ```julia

 inner(psi, Apply(H, psi))
 ```

 In this case, `Apply(H, psi)` represents the "lazy" evaluation of
 `apply(H, psi)`. The function `apply(H, psi)` performs the contraction of
 `H` with `psi` and then unprimes the results, so this versions ensures that
 the prime levels of the inner product will match.

 Although the new behavior seems less convenient, it makes it easier to
 generalize `inner(::MPS, ::MPO, ::MPS)` to other types of inputs, like `MPS`
 and `MPO` with different tag and prime conventions, multiple sites per tensor,
 `ITensor` inputs, etc.
 """
end

function deprecate_make_inds_match!(
  ::typeof(dot), ydag::MPS, A::MPO, x::MPS; make_inds_match::Bool=true
)
  N = length(x)
  if !hassameinds(siteinds, ydag, (A, x))
    sAx = siteinds((A, x))
    if any(s -> length(s) > 1, sAx)
      n = findfirst(n -> !hassameinds(siteinds(ydag, n), siteinds((A, x), n)), 1:N)
      error(
        """Calling `dot(ϕ::MPS, H::MPO, ψ::MPS)` with multiple site indices per MPO/MPS tensor but the site indices don't match. Even with `make_inds_match = true`, the case of multiple site indices per MPO/MPS is not handled automatically. The sites with unmatched site indices are:

            inds(ϕ[$n]) = $(inds(ydag[n]))

            inds(H[$n]) = $(inds(A[n]))

            inds(ψ[$n]) = $(inds(x[n]))

        Make sure the site indices of your MPO/MPS match. You may need to prime one of the MPS, such as `dot(ϕ', H, ψ)`.""",
      )
    end
    if !hassameinds(siteinds, ydag, (A, x)) && make_inds_match
      warn_once(inner_mps_mpo_mps_deprecation_warning(), :inner_mps_mpo_mps)
      replace_siteinds!(ydag, sAx)
    end
  end
  return ydag, A, x
end

function _log_or_not_dot(
  y::MPS, A::MPO, x::MPS, loginner::Bool; make_inds_match::Bool=true, kwargs...
)::Number
  N = length(A)
  check_hascommoninds(siteinds, A, x)
  ydag = dag(y)
  sim!(linkinds, ydag)
  ydag, A, x = deprecate_make_inds_match!(dot, ydag, A, x; make_inds_match)
  check_hascommoninds(siteinds, A, y)
  O = ydag[1] * A[1] * x[1]
  if loginner
    normO = norm(O)
    log_inner_tot = log(normO)
    O ./= normO
  end
  for j in 2:N
    O = O * ydag[j] * A[j] * x[j]
    if loginner
      normO = norm(O)
      log_inner_tot += log(normO)
      O ./= normO
    end
  end
  if loginner
    if !isreal(O[]) || real(O[]) < 0
      log_inner_tot += log(complex(O[]))
    end
    return log_inner_tot
  else
    return O[]
  end
end

"""
    dot(y::MPS, A::MPO, x::MPS)

Same as [`inner`](@ref).
"""
function dot(y::MPS, A::MPO, x::MPS; make_inds_match::Bool=true, kwargs...)
  return _log_or_not_dot(y, A, x, false; make_inds_match=make_inds_match, kwargs...)
end

"""
    logdot(B::MPO, y::MPS, A::MPO, x::MPS)
    Compute the logarithm of the inner product `⟨y|A|x⟩` efficiently and exactly.
    This is useful for larger MPS/MPO, where in the limit of large numbers of sites the inner product can diverge or approach zero.
    Same as [`loginner`](@ref).
"""
function logdot(y::MPS, A::MPO, x::MPS; make_inds_match::Bool=true, kwargs...)
  return _log_or_not_dot(y, A, x, true; make_inds_match=make_inds_match, kwargs...)
end

"""
    inner(y::MPS, A::MPO, x::MPS)

Compute `⟨y|A|x⟩ = ⟨y|Ax⟩` efficiently and exactly without making any intermediate
MPOs. In general it is more efficient and accurate than `inner(y, apply(A, x))`.

This is helpful for computing the expectation value of an operator `A`, which would be:

```julia
inner(x', A, x)
```

assuming `x` is normalized.

If you want to compute `⟨By|Ax⟩` you can use `inner(B::MPO, y::MPS, A::MPO, x::MPS)`.

This is helpful for computing the variance of an operator `A`, which would be:

```julia
inner(A, x, A, x) - inner(x', A, x) ^ 2
```

assuming `x` is normalized.

$(make_inds_match_docstring_warning())

Same as [`dot`](@ref).
"""
inner(y::MPS, A::MPO, x::MPS; kwargs...) = dot(y, A, x; kwargs...)

function inner(y::MPS, Ax::Apply{Tuple{MPO,MPS}})
  return inner(y', Ax.args[1], Ax.args[2])
end

"""
    loginner(y::MPS, A::MPO, x::MPS)
    Same as [`logdot`](@ref).
"""
loginner(y::MPS, A::MPO, x::MPS; kwargs...) = logdot(y, A, x; kwargs...)

"""
    dot(B::MPO, y::MPS, A::MPO, x::MPS)

Same as [`inner`](@ref).
"""
function dot(B::MPO, y::MPS, A::MPO, x::MPS; make_inds_match::Bool=true, kwargs...)::Number
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
  check_hascommoninds(siteinds, A, x)
  check_hascommoninds(siteinds, B, y)
  for j in eachindex(B)
    !hascommoninds(
      uniqueinds(siteinds(A, j), siteinds(x, j)), uniqueinds(siteinds(B, j), siteinds(y, j))
    ) && error(
      "$(typeof(x)) Ax and $(typeof(y)) By must share site indices. On site $j, Ax has site indices $(uniqueinds(siteinds(A, j), (siteinds(x, j)))) while By has site indices $(uniqueinds(siteinds(B, j), siteinds(y, j))).",
    )
  end
  ydag = dag(y)
  Bdag = dag(B)
  sim!(linkinds, ydag)
  sim!(linkinds, Bdag)
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
"""
    inner(B::MPO, y::MPS, A::MPO, x::MPS)

Compute `⟨By|A|x⟩ = ⟨By|Ax⟩` efficiently and exactly without making any intermediate
MPOs. In general it is more efficient and accurate than `inner(apply(B, y), apply(A, x))`.

This is helpful for computing the variance of an operator `A`, which would be:

```julia
inner(A, x, A, x) - inner(x, A, x) ^ 2
```

$(make_inds_match_docstring_warning())

Same as [`dot`](@ref).
"""
inner(B::MPO, y::MPS, A::MPO, x::MPS) = dot(B, y, A, x)

function dot(M1::MPO, M2::MPO; make_inds_match::Bool=false, kwargs...)
  if make_inds_match
    error("In dot(::MPO, ::MPO), make_inds_match is not currently supported")
  end
  return _log_or_not_dot(M1, M2, false; make_inds_match=make_inds_match)
end

# TODO: implement by combining the MPO indices and converting
# to MPS
function logdot(M1::MPO, M2::MPO; make_inds_match::Bool=false, kwargs...)
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
  iyax = dot(y', A, x; kwargs...)
  iaxax = dot(A, x, A, x; kwargs...)
  return sqrt(abs(1.0 + (iyy - 2 * real(iyax)) / iaxax))
end

error_contract(y::MPS, x::MPS, A::MPO) = error_contract(y, A, x)

"""
    apply(A::MPO, x::MPS; kwargs...)

Contract the `MPO` `A` with the `MPS` `x` and then map the prime level of the resulting
MPS back to 0.

Equivalent to `replaceprime(contract(A, x; kwargs...), 2 => 1)`.

See also [`contract`](@ref) for details about the arguments available.
"""
function apply(A::MPO, ψ::MPS; kwargs...)
  Aψ = contract(A, ψ; kwargs...)
  return replaceprime(Aψ, 1 => 0)
end

(A::MPO)(ψ::MPS; kwargs...) = apply(A, ψ; kwargs...)

Apply(A::MPO, ψ::MPS; kwargs...) = Applied(apply, (A, ψ), NamedTuple(kwargs))

function contract(A::MPO, ψ::MPS; alg="densitymatrix", kwargs...)
  if haskey(kwargs, :method)
    # Backwards compatibility, use `method`.
    alg = get(kwargs, :method, "densitymatrix")
  end

  # Keyword argument deprecations
  if alg == "DensityMatrix"
    @warn "In contract, method DensityMatrix is deprecated in favor of densitymatrix"
    alg = "densitymatrix"
  end
  if alg == "Naive"
    @warn "In contract, `alg=\"Naive\"` is deprecated in favor of `alg=\"naive\"`"
    alg = "naive"
  end

  return contract(Algorithm(alg), A, ψ; kwargs...)
end

contract_mpo_mps_doc = """
    contract(ψ::MPS, A::MPO; kwargs...) -> MPS
    *(::MPS, ::MPO; kwargs...) -> MPS

    contract(A::MPO, ψ::MPS; kwargs...) -> MPS
    *(::MPO, ::MPS; kwargs...) -> MPS

Contract the `MPO` `A` with the `MPS` `ψ`, returning an `MPS` with the unique
site indices of the `MPO`.

For example, for an MPO with site indices with prime levels of 1 and 0, such as
`-s'-A-s-`, and an MPS with site indices with prime levels of 0, such as
`-s-x`, the result is an MPS `y` with site indices with prime levels of 1,
`-s'-y = -s'-A-s-x`.

Since it is common to contract an MPO with prime levels of 1 and 0 with an MPS with
prime level of 0 and want a resulting MPS with prime levels of 0, we provide a
convenience function `apply`:
```julia
apply(A, x; kwargs...) = replaceprime(contract(A, x; kwargs...), 2 => 1)`.
```

Choose the method with the `method` keyword, for example
`"densitymatrix"` and `"naive"`.

# Keywords
- `cutoff::Float64=1e-13`: the cutoff value for truncating the density matrix
   eigenvalues. Note that the default is somewhat arbitrary and subject to
   change, in general you should set a `cutoff` value.
- `maxdim::Int=maxlinkdim(A) * maxlinkdim(ψ))`: the maximal bond dimension of the results MPS.
- `mindim::Int=1`: the minimal bond dimension of the resulting MPS.
- `normalize::Bool=false`: whether or not to normalize the resulting MPS.
- `method::String="densitymatrix"`: the algorithm to use for the contraction.
   Currently the options are "densitymatrix", where the network formed by the
   MPO and MPS is squared and contracted down to a density matrix which is
   diagonalized iteratively at each site, and "naive", where the MPO and MPS
   tensor are contracted exactly at each site and then a truncation of the
   resulting MPS is performed.

See also [`apply`](@ref).
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

function contract(::Algorithm"densitymatrix", A::MPO, ψ::MPS; kwargs...)::MPS
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

function _contract(::Algorithm"naive", A, ψ; kwargs...)
  truncate = get(kwargs, :truncate, true)

  A = sim(linkinds, A)
  ψ = sim(linkinds, ψ)

  N = length(A)
  if N != length(ψ)
    throw(DimensionMismatch("lengths of MPO ($N) and MPS ($(length(ψ))) do not match"))
  end

  ψ_out = typeof(ψ)(N)
  for j in 1:N
    ψ_out[j] = A[j] * ψ[j]
  end

  for b in 1:(N - 1)
    Al = commoninds(A[b], A[b + 1])
    ψl = commoninds(ψ[b], ψ[b + 1])
    l = [Al..., ψl...]
    if !isempty(l)
      C = combiner(l)
      ψ_out[b] *= C
      ψ_out[b + 1] *= dag(C)
    end
  end

  if truncate
    truncate!(ψ_out; kwargs...)
  end

  return ψ_out
end

function contract(alg::Algorithm"naive", A::MPO, ψ::MPS; kwargs...)
  return _contract(alg, A, ψ; kwargs...)
end

function contract(A::MPO, B::MPO; alg="zipup", kwargs...)
  return contract(Algorithm(alg), A, B; kwargs...)
end

function contract(alg::Algorithm"naive", A::MPO, B::MPO; kwargs...)
  return _contract(alg, A, B; kwargs...)
end

function contract(::Algorithm"zipup", A::MPO, B::MPO; kwargs...)
  if hassameinds(siteinds, A, B)
    error(
      "In `contract(A::MPO, B::MPO)`, MPOs A and B have the same site indices. The indices of the MPOs in the contraction are taken literally, and therefore they should only share one site index per site so the contraction results in an MPO. You may want to use `replaceprime(contract(A', B), 2 => 1)` or `apply(A, B)` which automatically adjusts the prime levels assuming the input MPOs have pairs of primed and unprimed indices.",
    )
  end
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
  R = ITensor(true)
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

"""
    apply(A::MPO, B::MPO; kwargs...)

Contract the `MPO` `A'` with the `MPO` `B` and then map the prime level of the resulting
MPO back to having pairs of indices with prime levels of 1 and 0.

Equivalent to `replaceprime(contract(A', B; kwargs...), 2 => 1)`.

See also [`contract`](@ref) for details about the arguments available.
"""
function apply(A::MPO, B::MPO; kwargs...)
  AB = contract(A', B; kwargs...)
  return replaceprime(AB, 2 => 1)
end

function apply(A1::MPO, A2::MPO, A3::MPO, As::MPO...; kwargs...)
  return apply(apply(A1, A2; kwargs...), A3, As...; kwargs...)
end

(A::MPO)(B::MPO; kwargs...) = apply(A, B; kwargs...)

contract_mpo_mpo_doc = """
    contract(A::MPO, B::MPO; kwargs...) -> MPO
    *(::MPO, ::MPO; kwargs...) -> MPO

Contract the `MPO` `A` with the `MPO` `B`, returning an `MPO` with the
site indices that are not shared between `A` and `B`.

If you are contracting two MPOs with the same sets of indices, likely you
want to call something like:

```julia
C = contract(A', B; cutoff=1e-12)
C = replaceprime(C, 2 => 1)
```

That is because if MPO `A` has the index structure `-s'-A-s-` and MPO `B`
has the Index structure `-s'-B-s-`, if we only want to contract over
on set of the indices, we would do `(-s'-A-s-)'-s'-B-s- = -s''-A-s'-s'-B-s- = -s''-C-s-`,
and then map the prime levels back to pairs of primed and unprimed indices with:
`replaceprime(-s''-C-s-, 2 => 1) = -s'-C-s-`.

Since this is a common use case, you can use the convenience function:

```julia
C = apply(A, B; cutoff=1e-12)
```

which is the same as the code above.

If you are contracting MPOs that have diverging norms, such as MPOs representing sums of local
operators, the truncation can become numerically unstable (see https://arxiv.org/abs/1909.06341 for
a more numerically stable alternative). For now, you can use the following options to contract
MPOs like that:

```julia
C = contract(A, B; alg="naive", truncate=false)
# Bring the indices back to pairs of primed and unprimed
C = apply(A, B; alg="naive", truncate=false)
```

# Keywords
- `cutoff::Float64=1e-14`: the cutoff value for truncating the density matrix
   eigenvalues. Note that the default is somewhat arbitrary and subject to change,
   in general you should set a `cutoff` value.
- `maxdim::Int=maxlinkdim(A) * maxlinkdim(B))`: the maximal bond dimension of the results MPS.
- `mindim::Int=1`: the minimal bond dimension of the resulting MPS.
- `alg="zipup"`: Either `"zipup"` or `"naive"`. `"zipup"` contracts pairs of
   site tensors and truncates with SVDs in a sweep across the sites, while `"naive"`
   first contracts pairs of tensor exactly and then truncates at the end if `truncate=true`.
- `truncate=true`: Enable or disable truncation. If `truncate=false`, ignore
   other truncation parameters like `cutoff` and `maxdim`. This is most relevant
   for the `"naive"` version, if you just want to contract the tensors pairwise
   exactly. This can be useful if you are contracting MPOs that have diverging
   norms, such as MPOs originating from sums of local operators.

See also [`apply`](@ref) for details about the arguments available.
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
  return sample(Random.default_rng(), M)
end

function sample(rng::AbstractRNG, M::MPO)
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
    r = rand(rng)
    # Will need n, An, and pn below
    n = 1
    projn = ITensor()
    pn = 0.0
    while n <= d
      projn = ITensor(s)
      projn[s => n] = 1.0
      pnc = (ρj * projn * prime(projn))[]
      if imag(pnc) > 1e-8
        @warn "In sample, probability $pnc is complex."
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
