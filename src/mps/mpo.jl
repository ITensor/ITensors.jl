
"""
    MPO

A finite size matrix product operator type. 
Keeps track of the orthogonality center.
"""
mutable struct MPO <: AbstractMPS
  data::Vector{ITensor}
  llim::Int
  rlim::Int
  function MPO(A::Vector{<:ITensor},
               llim::Int = 0,
               rlim::Int = length(A) + 1)
    new(A, llim, rlim)
  end
end

MPO() = MPO(ITensor[], 0, 0)

function MPO(::Type{ElT},
             sites::Vector{<:Index}) where {ElT <: Number}
  N = length(sites)
  v = Vector{ITensor}(undef, N)
  l = [Index(1, "Link,l=$ii") for ii in 1:N-1]
  for ii ∈ eachindex(sites)
    s = sites[ii]
    sp = prime(s)
    if ii == 1
      v[ii] = emptyITensor(ElT, s, sp, l[ii])
    elseif ii == N
      v[ii] = emptyITensor(ElT, l[ii-1], s, sp)
    else
      v[ii] = emptyITensor(ElT, l[ii-1], s, sp, l[ii])
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
function MPO(::Type{ElT},
             sites::Vector{<:Index},
             ops::Vector{String}) where {ElT <: Number}
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
      this_it = emptyITensor(ElT, links[ii], si, si')
      for jj in 1:d, jjp in 1:d
        this_it[links[ii](1), si[jj], si'[jjp]] = spin_op[si[jj], si'[jjp]]
      end
    elseif ii == N
      this_it = emptyITensor(ElT, links[ii-1], si, si')
      for jj in 1:d, jjp in 1:d
        this_it[links[ii-1](1), si[jj], si'[jjp]] = spin_op[si[jj], si'[jjp]]
      end
    else
      this_it = emptyITensor(ElT, links[ii-1], links[ii], si, si')
      for jj in 1:d, jjp in 1:d
        this_it[links[ii-1](1),
                links[ii](1),
                si[jj],
                si'[jjp]] = spin_op[si[jj], si'[jjp]]
      end
    end
    its[ii] = this_it
  end
  MPO(its)
end

MPO(sites::Vector{<:Index},
    ops::Vector{String}) = MPO(Float64, sites, ops)

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

MPO(sites::Vector{<:Index},
    op::String) = MPO(Float64, sites, op)

function randomMPO(sites::Vector{<:Index}, m::Int=1)
  M = MPO(sites, "Id")
  for i ∈ eachindex(sites)
    randn!(M[i])
    normalize!(M[i])
  end
  m > 1 && throw(ArgumentError("randomMPO: currently only m==1 supported"))
  return M
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
function LinearAlgebra.dot(y::MPS, A::MPO, x::MPS;
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
function Base.:*(A::MPO, psi::MPS; kwargs...)::MPS
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
    return _contract_densitymatrix(A, psi; kwargs...)
  elseif method == "naive"
    return _contract_naive(A, psi; kwargs...)
  end
  throw(ArgumentError("Method $method not supported"))
end

Base.:*(psi::MPS, A::MPO; kwargs...) = *(A, psi; kwargs...)

function _contract_densitymatrix(A::MPO, psi::MPS; kwargs...)::MPS
  n = length(A)
  n != length(psi) && throw(DimensionMismatch("lengths of MPO ($n) and MPS ($(length(psi))) do not match"))
  psi_out         = similar(psi)
  cutoff::Float64 = get(kwargs, :cutoff, 1e-13)
  maxdim::Int     = get(kwargs,:maxdim,maxlinkdim(psi))
  mindim::Int     = max(get(kwargs,:mindim,1), 1)
  normalize::Bool = get(kwargs, :normalize, false) 

  any(i -> isnothing(i), common_siteinds(A, psi)) && error("In `contract(A::MPO, x::MPS)`, `A` and `x` must share a set of site indices")

  # In case A and psi have the same link indices
  A = sim_linkinds(A)

  psi_c = dag(psi)
  A_c = dag(A)

  # To not clash with the link indices of A and psi
  sim_linkinds!(A_c)
  sim_linkinds!(psi_c)
  sim_common_siteinds!(A_c, psi_c)

  # A version helpful for making the density matrix
  simA_c = sim_unique_siteinds(A_c, psi_c)

  # Store the left environment tensors
  E = Vector{ITensor}(undef, n-1)

  E[1] = psi[1] * A[1] * A_c[1] * psi_c[1]
  for j in 2:n-1
    E[j] = E[j-1] * psi[j] * A[j] * A_c[j] * psi_c[j]
  end
  R = psi[n] * A[n]
  simR_c = psi_c[n] * simA_c[n]
  ρ = E[n-1] * R * simR_c
  l = linkind(psi, n-1)
  ts = isnothing(l) ? "" : tags(l)
  s = unique_siteind(A, psi, n)
  s̃ = unique_siteind(simA_c, psi_c, n)
  Lis = IndexSet(s)
  Ris = IndexSet(s̃)
  F = eigen(ρ, Lis, Ris; ishermitian=true, 
                         tags=ts, 
                         kwargs...)
  D, U, Ut = F.D, F.V, F.Vt
  l_renorm, r_renorm = F.l, F.r
  psi_out[n] = Ut
  R = R * dag(Ut) * psi[n-1] * A[n-1]
  simR_c = simR_c * U * psi_c[n-1] * simA_c[n-1]
  for j in reverse(2:n-1)
    s = unique_siteind(A, psi, j)
    s̃ = unique_siteind(simA_c, psi_c, j)
    ρ = E[j-1] * R * simR_c
    l = linkind(psi, j-1)
    ts = isnothing(l) ? "" : tags(l)
    Lis = IndexSet(s, l_renorm)
    Ris = IndexSet(s̃, r_renorm)
    F = eigen(ρ, Lis, Ris; ishermitian=true,
                           tags=ts, 
                           kwargs...)
    D, U, Ut = F.D, F.V, F.Vt
    l_renorm, r_renorm = F.l, F.r
    psi_out[j] = Ut
    R = R * dag(Ut) * psi[j-1] * A[j-1]
    simR_c = simR_c * U * psi_c[j-1] * simA_c[j-1]
  end
  if normalize
    R ./= norm(R)
  end
  psi_out[1] = R
  setleftlim!(psi_out, 0)
  setrightlim!(psi_out, 2)
  return psi_out
end

function _contract_naive(A::MPO, psi::MPS; kwargs...)::MPS
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
    new_ci = sim(ci)
    replaceind!(res[i], ci, new_ci)
    replaceind!(res[i+1], ci, new_ci)
    @assert commonind(res[i], res[i+1]) != commonind(A[i], A[i+1])
  end
  sites_A = Index[]
  sites_B = Index[]
  for (AA, BB) in zip(data(A_), data(B_))
    sda = setdiff(inds(AA, "Site"), inds(BB, "Site"))
    sdb = setdiff(inds(BB, "Site"), inds(AA, "Site"))
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
