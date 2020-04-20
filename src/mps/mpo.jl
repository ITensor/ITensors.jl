
"""
    MPO

A finite size matrix product operator type. 
Keeps track of the orthogonality center.
"""
mutable struct MPO <: AbstractMPS
  length::Int
  data::Vector{ITensor}
  llim::Int
  rlim::Int
  function MPO(N::Int,
               A::Vector{<:ITensor},
               llim::Int = 0,
               rlim::Int = N+1)
    new(N, A, llim, rlim)
  end
end

MPO() = MPO(0, Vector{ITensor}(), 0, 0)

function MPO(sites::Vector{<:Index})
  N = length(sites)
  v = Vector{ITensor}(undef, N)
  l = [Index(1, "Link,l=$ii") for ii in 1:N-1]
  for ii ∈ eachindex(sites)
    s = sites[ii]
    sp = prime(s)
    if ii == 1
      v[ii] = ITensor(s, sp, l[ii])
    elseif ii == N
      v[ii] = ITensor(l[ii-1], s, sp)
    else
      v[ii] = ITensor(l[ii-1], s, sp, l[ii])
    end
  end
  return MPO(N, v)
end
 
MPO(A::Vector{<:ITensor}) = MPO(length(A), A)

"""
    MPO(N::Int)

Make an MPO of length `N` filled with default ITensors.
"""
MPO(N::Int) = MPO(N, fill(ITensor(), N))

"""
    MPO(sites, ops::Vector{String})

Make an MPO with pairs of sites `s[i]` and `s[i]'`
and operators `ops` on each site.
"""
function MPO(sites,
             ops::Vector{String})
  N = length(sites)
  its = Vector{ITensor}(undef, N)
  links = Vector{Index}(undef, N)
  for ii ∈ eachindex(sites)
    si = sites[ii]
    d = dim(si)
    spin_op = op(sites, ops[ii], ii)
    links[ii] = Index(1, "Link,n=$ii")
    local this_it
    if ii == 1
      this_it = ITensor(links[ii], si, si')
      for jj in 1:d, jjp in 1:d
        this_it[links[ii](1), si[jj], si'[jjp]] = spin_op[si[jj], si'[jjp]]
      end
    elseif ii == N
      this_it = ITensor(links[ii-1], si, si')
      for jj in 1:d, jjp in 1:d
        this_it[links[ii-1](1), si[jj], si'[jjp]] = spin_op[si[jj], si'[jjp]]
      end
    else
      this_it = ITensor(links[ii-1], links[ii], si, si')
      for jj in 1:d, jjp in 1:d
        this_it[links[ii-1](1),
                links[ii](1),
                si[jj],
                si'[jjp]] = spin_op[si[jj], si'[jjp]]
      end
    end
    its[ii] = this_it
  end
  MPO(N, its)
end

"""
    MPO(sites, ops::Vector{String})

Make an MPO with pairs of sites `s[i]` and `s[i]'`
and operators `ops` on each site.
"""
MPO(sites, ops::String) = MPO(sites, fill(ops, length(sites)))

function randomMPO(sites, m::Int=1)
  M = MPO(sites)
  for i ∈ eachindex(sites)
    randn!(M[i])
    normalize!(M[i])
  end
  m > 1 && throw(ArgumentError("randomMPO: currently only m==1 supported"))
  return M
end

"""
    siteind(A::MPO, x::MPS, j::Int)

Get the site index of MPO `A` that is unique to
`A` (not shared with MPS `x`).
"""
function siteind(A::MPO, x::MPS, j::Int)
  N = length(A)
  if j == 1
    si = uniqueind(A[j], A[j+1], x[j])
  elseif j == N
    si = uniqueind(A[j], A[j-1], x[j])
  else
    si = uniqueind(A[j], A[j-1], A[j+1], x[j])
  end
  return si
end

"""
    siteinds(A::MPO, x::MPS)

Get the site indices of MPO `A` that are unique to
`A` (not shared with MPS `x`), as a `Vector{<:Index}`.

This is the same as getting the `siteinds` of `A|x>`, i.e.
`siteinds(A * x)`, without doing the contraction.
"""
siteinds(A::MPO, x::MPS) = [siteind(A, x, j) for j in eachindex(A)]

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
function LinearAlgebra.dot(y::MPS, A::MPO, x::MPS;
                           make_inds_match::Bool = true)::Number
  N = length(A)
  if length(y) != N || length(x) != N
      throw(DimensionMismatch("inner: mismatched lengths $N and $(length(x)) or $(length(y))"))
  end
  ydag = dag(y)
  simlinkinds!(ydag)
  if make_inds_match
    sAx = siteinds(A, x)
    replacesiteinds!(ydag, sAx)
  end
  O = ydag[1]*A[1]*x[1]
  for j in 2:N
    O = O*ydag[j]*A[j]*x[j]
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
function LinearAlgebra.dot(B::MPO, y::MPS,
                           A::MPO, x::MPS;
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
    ABcommon = uniqueind(inds(A[j], "Site"), IndexSet(Axcommon))
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

inner(B::MPO, y::MPS, A::MPO, x::MPS) = dot(B, y, A, x)

"""
    error_mul(y::MPS, A::MPO, x::MPS; make_inds_match::Bool = true)
    error_mul(y::MPS, x::MPS, x::MPO; make_inds_match::Bool = true)

Compute the distance between A|x> and an approximation MPS y:
`| |y> - A|x> |/| A|x> | = √(1 + (<y|y> - 2*real(<y|A|x>))/<Ax|A|x>)`.

If `make_inds_match = true`, the function attempts match the site 
indices of `y` with the site indices of `A` that are not common
with `x`.
"""
function error_mul(y::MPS, A::MPO, x::MPS; kwargs...)
  N = length(A)
  if length(y) != N || length(x) != N
    throw(DimensionMismatch("inner: mismatched lengths $N and $(length(x)) or $(length(y))"))
  end
  iyy   = dot(y, y; kwargs...)
  iyax  = dot(y, A, x; kwargs...)
  iaxax = dot(A, x, A, x; kwargs...)
  return sqrt(abs(1. + (iyy - 2*real(iyax))/iaxax))
end

error_mul(y::MPS, x::MPS, A::MPO) = error_mul(y, A, x)

"""
    mul(::MPS, ::MPO; kwargs...)
    *(::MPS, ::MPO; kwargs...)

    mul(::MPO, ::MPS; kwargs...)
    *(::MPO, ::MPS; kwargs...)

Contract the MPO with the MPS, returning an MPS with the unique
site indices of the MPO.

Choose the method with the `method` keyword, for example
"densitymatrix" and "naive".
"""
function Base.:*(A::MPO, psi::MPS; kwargs...)::MPS
  method = get(kwargs, :method, "densitymatrix")
  if method == "DensityMatrix"
    @warn "In mul, method DensityMatrix is deprecated in favor of densitymatrix"
    method = "densitymatrix"
  end
  if method == "densitymatrix"
    return _mul_densitymatrix(A, psi; kwargs...)
  elseif method == "naive" || method == "Naive"
    return _mul_naive(A, psi; kwargs...)
  end
  throw(ArgumentError("Method $method not supported"))
end

Base.:*(psi::MPS, A::MPO; kwargs...) = *(A, psi; kwargs...)

function _mul_densitymatrix(A::MPO, psi::MPS; kwargs...)::MPS
  n = length(A)
  n != length(psi) && throw(DimensionMismatch("lengths of MPO ($n) and MPS ($(length(psi))) do not match"))
  psi_out         = similar(psi)
  cutoff::Float64 = get(kwargs, :cutoff, 1e-13)
  maxdim::Int     = get(kwargs,:maxdim,maxlinkdim(psi))
  mindim::Int     = max(get(kwargs,:mindim,1), 1)
  normalize::Bool = get(kwargs, :normalize, false) 
  all(x -> x != Index(),
      [siteind(A, psi, j) for j in 1:n]) || 
  throw(ErrorException("MPS and MPO have different site indices in mul method 'densitymatrix'"))

  rand_plev = 14741
  psi_c     = dag(copy(psi))
  A_c       = dag(copy(A))
  prime!(psi_c, rand_plev)
  prime!(A_c, rand_plev)
  for j in 1:n
    s = siteind(A, psi, j)
    s_dag = siteind(A_c, psi_c, j)
    replaceind!(A_c[j], s_dag, s)
  end
  E = Vector{ITensor}(undef, n-1)
  E[1] = psi[1]*A[1]*A_c[1]*psi_c[1]
  for j in 2:n-1
    E[j] = E[j-1]*psi[j]*A[j]*A_c[j]*psi_c[j]
  end
  O     = psi[n] * A[n]
  ρ     = E[n-1] * O * dag(prime(O, rand_plev))
  ts    = tags(commonind(psi[n], psi[n-1]))
  Lis   = commonind(ρ, A[n])
  Ris   = prime(Lis, rand_plev)
  FU, D = eigen(ρ, Lis, Ris; ishermitian=true, 
                             tags=ts, 
                             kwargs...)
  psi_out[n] = dag(FU)
  O     = O * FU * psi[n-1] * A[n-1]
  for j in reverse(2:n-1)
    dO  = prime(dag(O), rand_plev)
    ρ   = E[j-1] * O * dO
    ts  = tags(commonind(psi[j], psi[j-1]))
    Lis = IndexSet(commonind(ρ, A[j]),
                   commonind(ρ, psi_out[j+1])) 
    Ris = prime(Lis, rand_plev)
    FU, D = eigen(ρ, Lis, Ris; ishermitian=true,
                               tags=ts, 
                               kwargs...)
    psi_out[j] = dag(FU)
    O = O * FU * psi[j-1] * A[j-1]
  end
  if normalize
    O /= norm(O)
  end
  psi_out[1]    = copy(O)
  setleftlim!(psi_out, 0)
  setrightlim!(psi_out, 2)
  return psi_out
end

function _mul_naive(A::MPO, psi::MPS; kwargs...)::MPS
  truncate = get(kwargs,:truncate,true)

  N = length(A)
  if N != length(psi) 
    throw(DimensionMismatch("lengths of MPO ($N) and MPS ($(length(psi))) do not match"))
  end

  psi_out = MPS(N)
  for j=1:N
    psi_out[j] = A[j]*psi[j]
  end

  for b=1:(N-1)
    Al = commonind(A[b],A[b+1])
    pl = commonind(psi[b],psi[b+1])
    C = combiner(Al,pl)
    psi_out[b] *= C
    psi_out[b+1] *= dag(C)
  end

  if truncate
    truncate!(psi_out;kwargs...)
  end

  return psi_out
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

  links_A = inds.(A.data, "Link")
  links_B = inds.(B.data, "Link")

  res = deepcopy(A_)
  for i in 1:N-1
    ci = commonind(res[i], res[i+1])
    new_ci = Index(dim(ci), tags(ci))
    replaceind!(res[i], ci, new_ci)
    replaceind!(res[i+1], ci, new_ci)
    @assert commonind(res[i], res[i+1]) != commonind(A[i], A[i+1])
  end
  sites_A = Index[]
  sites_B = Index[]
  for (AA, BB) in zip(data(A_), data(B_))
    sda = setdiff(inds(AA, "Site"), inds(BB, "Site"))
    sdb = setdiff(inds(BB, "Site"), inds(AA, "Site"))
    length(sda) != 1 && error("In mul(::MPO, ::MPO), MPOs must have exactly one shared site index")
    length(sdb) != 1 && error("In mul(::MPO, ::MPO), MPOs must have exactly one shared site index")
    push!(sites_A, sda[1])
    push!(sites_B, sdb[1])
  end
  res[1] = ITensor(sites_A[1], sites_B[1], commonind(res[1], res[2]))
  for i in 1:N-2
    if i == 1
      clust = A_[i] * B_[i]
    else
      clust = nfork * A_[i] * B_[i]
    end
    lA = commonind(A_[i], A_[i+1])
    lB = commonind(B_[i], B_[i+1])
    nfork = ITensor(lA, lB, commonind(res[i], res[i+1]))
    res[i], nfork = factorize(clust,
                              inds(res[i]),
                              ortho="left",
                              tags=tags(lA),
                              cutoff=cutoff,
                              maxdim=maxdim,
                              mindim=mindim)
    mid = dag(commonind(res[i], nfork))
    res[i+1] = ITensor(mid,
                       sites_A[i+1],
                       sites_B[i+1],
                       commonind(res[i+1], res[i+2]))
  end
  clust = nfork * A_[N-1] * B_[N-1]
  nfork = clust * A_[N] * B_[N]

  # in case we primed A
  A_ind = uniqueind(inds(A_[N-1], "Site"), inds(B_[N-1], "Site"))
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

