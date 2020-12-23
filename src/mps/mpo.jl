
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

function MPO(A::Vector{<:ITensor};
             ortho_lims::UnitRange = 1:length(A))
  return MPO(A, first(ortho_lims)-1, last(ortho_lims)+1)
end

MPO() = MPO(ITensor[], 0, 0)

function convert(::Type{MPS}, M::MPO)
  return MPS(data(M); ortho_lims = ortho_lims(M))
end

function convert(::Type{MPO}, M::MPS)
  return MPO(data(M); ortho_lims = ortho_lims(M))
end

function MPO(::Type{ElT},
             sites::Vector{<:Index}) where {ElT <: Number}
  N = length(sites)
  v = Vector{ITensor}(undef, N)
  if N == 0
    return MPO()
  elseif N == 1
    v[1] = emptyITensor(ElT, dag(sites[1]), sites[1]')
    return MPO(v)
  end
  space_ii = all(hasqns, sites) ? [QN() => 1] : 1
  l = [Index(space_ii, "Link,l=$ii") for ii in 1:N-1]
  for ii ∈ eachindex(sites)
    s = sites[ii]
    if ii == 1
      v[ii] = emptyITensor(ElT, dag(s), s', l[ii])
    elseif ii == N
      v[ii] = emptyITensor(ElT, dag(l[ii-1]), dag(s), s')
    else
      v[ii] = emptyITensor(ElT, dag(l[ii-1]), dag(s), s', l[ii])
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
function MPO(::Type{ElT}, sites::Vector{<:Index},
             ops::Vector{String}) where {ElT <: Number}
  N = length(sites)
  ampo = AutoMPO() + [ops[n] => n for n in 1:N]
  M = MPO(ampo, sites)

  # Currently, AutoMPO does not output the optimally truncated
  # MPO (see https://github.com/ITensor/ITensors.jl/issues/526)
  # So here, we need to first normalize, then truncate, then
  # return the normalization.
  lognormM = lognorm(M)
  M ./= exp(lognormM / N)
  truncate!(M; cutoff = 1e-15)
  M .*= exp(lognormM / N)
  return M
end

function MPO(::Type{ElT}, sites::Vector{<:Index},
             fops::Function) where {ElT <: Number}
  ops = [fops(n) for n in 1:length(sites)]
  return MPO(ElT, sites, ops)
end

MPO(sites::Vector{<:Index}, ops) = MPO(Float64, sites, ops)

"""
    MPO([::Type{ElT} = Float64, ]sites, op::String)

Make an MPO with pairs of sites `s[i]` and `s[i]'`
and operator `op` on every site.
"""
function MPO(::Type{ElT},
             sites::Vector{<:Index},
             op::String) where {ElT <: Number}
  return MPO(ElT, sites, fill(op, length(sites)))
end

MPO(sites::Vector{<:Index}, op::String) = MPO(Float64, sites, op)

function randomMPO(sites::Vector{<:Index}, m::Int=1)
  M = MPO(sites, "Id")
  for i ∈ eachindex(sites)
    randn!(M[i])
    normalize!(M[i])
  end
  m > 1 && throw(ArgumentError("randomMPO: currently only m==1 supported"))
  return M
end

MPO(A::ITensor, sites::Vector{<:Index}; kwargs...) =
  MPO(A, IndexSet.(prime.(sites), dag.(sites)); kwargs...)

"""
    MPO(A::MPS; kwargs...)

For an MPS `|A>`, make the MPO `|A><A|`.
Keyword arguments like `cutoff` can be used to
truncate the resulting MPO.
"""
function MPO(A::MPS; kwargs...)
  N = length(A)
  M = MPO([A[n]' * dag(A[n]) for n in 1:N])
  truncate!(M; kwargs...)
  return M
end

# XXX: rename originalsiteind?
"""
    siteind(M::MPO, j::Int; plev = 0, kwargs...)

Get the first site Index of the MPO found, by
default with prime level 0. 
"""
siteind(M::MPO, j::Int; kwargs...) = siteind(first, M, j; plev = 0, kwargs...)

# TODO: make this return the site indices that would have
# been used to create the MPO? I.e.:
# [dag(siteinds(M, j; plev = 0, kwargs...)) for j in 1:length(M)]
"""
    siteinds(M::MPO; kwargs...)

Get a Vector of IndexSets of all the site indices of M.
"""
siteinds(M::MPO; kwargs...) = siteinds(all, M; kwargs...)

# XXX: rename originalsiteinds?
"""
    firstsiteinds(M::MPO; kwargs...)

Get a Vector of the first site Index found on each site of M.

By default, it finds the first site Index with prime level 0.
"""
firstsiteinds(M::MPO; kwargs...) = siteinds(first, M; plev = 0, kwargs...)

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
function dot(y::MPS, A::MPO, x::MPS;
             make_inds_match::Bool = true)::Number
  N = length(A)
  if length(y) != N || length(x) != N
    throw(DimensionMismatch("inner: mismatched lengths $N and $(length(x)) or $(length(y))"))
  end
  ydag = dag(y)
  sim_linkinds!(ydag)
  if make_inds_match
    sAx = unique_siteinds(A, x)
    replace_siteinds!(ydag, sAx)
  end
  O = ydag[1] * A[1] * x[1]
  for j in 2:N
    O = O * ydag[j] * A[j] * x[j]
  end
  return O[]
end

inner(y::MPS, A::MPO, x::MPS) = dot(y, A, x)

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
function dot(B::MPO, y::MPS, A::MPO, x::MPS;
             make_inds_match::Bool = true)::Number
  !make_inds_match && error("make_inds_match = false not currently supported in dot(::MPO, ::MPS, ::MPO, ::MPS)")
  N = length(B)
  if length(y) != N || length(x) != N || length(A) != N
      throw(DimensionMismatch("inner: mismatched lengths $N and $(length(x)) or $(length(y)) or $(length(A))"))
  end
  ydag = dag(y)
  prime!(ydag, 2)
  Bdag = dag(B)
  prime!(Bdag)
  # Swap prime levels 1 -> 2 and 2 -> 1.
  for j in eachindex(Bdag)
    Axcommon = commonind(A[j], x[j])
    ABcommon = uniqueind(filterinds(A[j]; tags = "Site"), IndexSet(Axcommon))
    swapprime!(Bdag[j],2,3)
    swapprime!(Bdag[j],1,2)
    swapprime!(Bdag[j],3,1)
    noprime!(Bdag[j],prime(ABcommon,2))
  end
  yB = ydag[1] * Bdag[1]
  Ax = A[1] * x[1]
  O = yB*Ax
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

function dot(M1::MPO, M2::MPO;
             make_inds_match::Bool = false)
  if make_inds_match
    error("In dot(::MPO, ::MPO), make_inds_match is not currently supported")
  end
  return _log_or_not_dot(M1, M2, false;
                         make_inds_match = make_inds_match)
end

# TODO: implement by combing the MPO indices and converting
# to MPS
function logdot(M1::MPO, M2::MPO;
                make_inds_match::Bool = false)
  if make_inds_match
    error("In dot(::MPO, ::MPO), make_inds_match is not currently supported")
  end
  return _log_or_not_dot(M1, M2, true;
                         make_inds_match = make_inds_match)
end

function tr(M::MPO; plev::Pair{Int, Int} = 0 => 1,
            tags::Pair = ts"" => ts"")
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
  L = tr(M[1]; plev = plev, tags = tags)
  for j in 2:N
    L *= M[j]
    L = tr(L; plev = plev, tags = tags)
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
    throw(DimensionMismatch("inner: mismatched lengths $N and $(length(x)) or $(length(y))"))
  end
  iyy   = dot(y, y; kwargs...)
  iyax  = dot(y, A, x; kwargs...)
  iaxax = dot(A, x, A, x; kwargs...)
  return sqrt(abs(1. + (iyy - 2*real(iyax))/iaxax))
end

error_contract(y::MPS, x::MPS, A::MPO) = error_contract(y, A, x)

"""
    contract(::MPS, ::MPO; kwargs...)
    *(::MPS, ::MPO; kwargs...)

    contract(::MPO, ::MPS; kwargs...)
    *(::MPO, ::MPS; kwargs...)

Contract the MPO with the MPS, returning an MPS with the unique
site indices of the MPO.

Choose the method with the `method` keyword, for example
"densitymatrix" and "naive".
"""
function *(A::MPO, ψ::MPS; kwargs...)
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

Base.:*(ψ::MPS, A::MPO; kwargs...) = *(A, ψ; kwargs...)

function _contract_densitymatrix(A::MPO, ψ::MPS; kwargs...)::MPS
  n = length(A)
  n != length(ψ) && throw(DimensionMismatch("lengths of MPO ($n) and MPS ($(length(ψ))) do not match"))
  ψ_out         = similar(ψ)
  cutoff::Float64 = get(kwargs, :cutoff, 1e-13)
  maxdim::Int     = get(kwargs,:maxdim,maxlinkdim(ψ))
  mindim::Int     = max(get(kwargs,:mindim,1), 1)
  normalize::Bool = get(kwargs, :normalize, false) 

  any(i -> isnothing(i), common_siteinds(A, ψ)) && error("In `contract(A::MPO, x::MPS)`, `A` and `x` must share a set of site indices")

  # In case A and ψ have the same link indices
  A = sim_linkinds(A)

  ψ_c = dag(ψ)
  A_c = dag(A)

  # To not clash with the link indices of A and ψ
  sim_linkinds!(A_c)
  sim_linkinds!(ψ_c)
  sim_common_siteinds!(A_c, ψ_c)

  # A version helpful for making the density matrix
  simA_c = sim_unique_siteinds(A_c, ψ_c)

  # Store the left environment tensors
  E = Vector{ITensor}(undef, n-1)

  E[1] = ψ[1] * A[1] * A_c[1] * ψ_c[1]
  for j in 2:n-1
    E[j] = E[j-1] * ψ[j] * A[j] * A_c[j] * ψ_c[j]
  end
  R = ψ[n] * A[n]
  simR_c = ψ_c[n] * simA_c[n]
  ρ = E[n-1] * R * simR_c
  l = linkind(ψ, n-1)
  ts = isnothing(l) ? "" : tags(l)
  s = unique_siteind(A, ψ, n)
  s̃ = unique_siteind(simA_c, ψ_c, n)
  Lis = IndexSet(s)
  Ris = IndexSet(s̃)
  F = eigen(ρ, Lis, Ris; ishermitian=true, 
                         tags=ts, 
                         kwargs...)
  D, U, Ut = F.D, F.V, F.Vt
  l_renorm, r_renorm = F.l, F.r
  ψ_out[n] = Ut
  R = R * dag(Ut) * ψ[n-1] * A[n-1]
  simR_c = simR_c * U * ψ_c[n-1] * simA_c[n-1]
  for j in reverse(2:n-1)
    s = unique_siteind(A, ψ, j)
    s̃ = unique_siteind(simA_c, ψ_c, j)
    ρ = E[j-1] * R * simR_c
    l = linkind(ψ, j-1)
    ts = isnothing(l) ? "" : tags(l)
    Lis = IndexSet(s, l_renorm)
    Ris = IndexSet(s̃, r_renorm)
    F = eigen(ρ, Lis, Ris; ishermitian=true,
                           tags=ts, 
                           kwargs...)
    D, U, Ut = F.D, F.V, F.Vt
    l_renorm, r_renorm = F.l, F.r
    ψ_out[j] = Ut
    R = R * dag(Ut) * ψ[j-1] * A[j-1]
    simR_c = simR_c * U * ψ_c[j-1] * simA_c[j-1]
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
  truncate = get(kwargs,:truncate,true)

  N = length(A)
  if N != length(ψ) 
    throw(DimensionMismatch("lengths of MPO ($N) and MPS ($(length(ψ))) do not match"))
  end

  ψ_out = MPS(N)
  for j=1:N
    ψ_out[j] = A[j]*ψ[j]
  end

  for b=1:(N-1)
    Al = commonind(A[b],A[b+1])
    pl = commonind(ψ[b],ψ[b+1])
    C = combiner(Al,pl)
    ψ_out[b] *= C
    ψ_out[b+1] *= dag(C)
  end

  if truncate
    truncate!(ψ_out;kwargs...)
  end

  return ψ_out
end

function Base.:*(A::MPO, B::MPO; kwargs...)
  cutoff::Float64 = get(kwargs, :cutoff, 1e-14)
  resp_degen::Bool = get(kwargs, :respect_degenerate, true)
  maxdim::Int = get(kwargs,:maxdim,maxlinkdim(A)*maxlinkdim(B))
  mindim::Int = max(get(kwargs,:mindim,1), 1)
  N = length(A)
  N != length(B) && throw(DimensionMismatch("lengths of MPOs A ($N) and B ($(length(B))) do not match"))
  A_ = copy(A)
  orthogonalize!(A_, 1)
  B_ = copy(B)
  orthogonalize!(B_, 1)

  links_A = filterinds.(A.data; tags = "Link")
  links_B = filterinds.(B.data; tags = "Link")

  res = deepcopy(A_)
  for i in 1:N-1
    ci = commonind(res[i], res[i+1])
    new_ci = sim(ci)
    replaceind!(res[i], ci, new_ci)
    replaceind!(res[i+1], ci, new_ci)
    @assert commonind(res[i], res[i+1]) != commonind(A[i], A[i+1])
  end
  sites_A = Index[]
  sites_B = Index[]
  for (AA, BB) in zip(data(A_), data(B_))
    sda = setdiff(filterinds(AA; tags="Site"), filterinds(BB; tags="Site"))
    sdb = setdiff(filterinds(BB; tags="Site"), filterinds(AA; tags="Site"))
    length(sda) != 1 && error("In contract(::MPO, ::MPO), MPOs must have exactly one shared site index")
    length(sdb) != 1 && error("In contract(::MPO, ::MPO), MPOs must have exactly one shared site index")
    push!(sites_A, sda[1])
    push!(sites_B, sdb[1])
  end
  res[1] = emptyITensor(sites_A[1], sites_B[1], commonind(res[1], res[2]))
  for i in 1:N-2
    if i == 1
      clust = A_[i] * B_[i]
    else
      clust = nfork * A_[i] * B_[i]
    end
    lA = commonind(A_[i], A_[i+1])
    lB = commonind(B_[i], B_[i+1])
    nfork = emptyITensor(lA, lB, commonind(res[i], res[i+1]))
    res[i], nfork = factorize(clust,
                              inds(res[i]),
                              ortho="left",
                              tags=tags(lA),
                              cutoff=cutoff,
                              maxdim=maxdim,
                              mindim=mindim)
    mid = dag(commonind(res[i], nfork))
    res[i+1] = emptyITensor(mid,
                           sites_A[i+1],
                           sites_B[i+1],
                           commonind(res[i+1], res[i+2]))
  end
  clust = nfork * A_[N-1] * B_[N-1]
  nfork = clust * A_[N] * B_[N]

  # in case we primed A
  A_ind = uniqueind(filterinds(A_[N-1]; tags = "Site"),
                    filterinds(B_[N-1]; tags = "Site"))
  Lis = IndexSet(A_ind, sites_B[N-1], commonind(res[N-2], res[N-1]))
  U, V = factorize(nfork, Lis; 
                   ortho="right",
                   cutoff=cutoff,
                   tags="Link,n=$(N-1)",
                   maxdim=maxdim,
                   mindim=mindim)
  res[N-1] = U
  res[N] = V
  truncate!(res;kwargs...)
  return res
end

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
  for n in reverse(1:N-1)
    R[n] = M[n] * δ(dag(s[n])) * R[n+1]
  end

  if abs(1.0-R[1][]) > 1E-8
    error("sample: MPO is not normalized, norm=$(norm(M[1]))")
  end

  result = zeros(Int,N)
  ρj = M[1] * R[2]
  Lj = ITensor()

  for j in 1:N
    s = siteind(M,j)
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
      if j == N-1
        ρj = Lj * M[j+1]
      else
        ρj = Lj * M[j+1] * R[j+2]
      end
      s = siteind(M, j+1)
      normj = (ρj * δ(s', s))[]
      ρj ./= normj
    end
  end
  return result
end

function HDF5.write(parent::Union{HDF5File,HDF5Group},
                    name::AbstractString,
                    M::MPO)
  g = g_create(parent,name)
  attrs(g)["type"] = "MPO"
  attrs(g)["version"] = 1
  N = length(M)
  write(g, "rlim", M.rlim)
  write(g, "llim", M.llim)
  write(g, "length", N)
  for n=1:N
    write(g,"MPO[$(n)]", M[n])
  end
end

function HDF5.read(parent::Union{HDF5File,HDF5Group},
                   name::AbstractString,
                   ::Type{MPO})
  g = g_open(parent,name)
  if read(attrs(g)["type"]) != "MPO"
    error("HDF5 group or file does not contain MPO data")
  end
  N = read(g, "length")
  rlim = read(g, "rlim")
  llim = read(g, "llim")
  v = [read(g,"MPO[$(i)]",ITensor) for i in 1:N]
  return MPO( v, llim, rlim)
end

