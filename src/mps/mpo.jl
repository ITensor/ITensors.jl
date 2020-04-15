
mutable struct MPO <: AbstractMPS
  length::Int
  data::Vector{ITensor}
  llim::Int
  rlim::Int

  MPO() = new(0,Vector{ITensor}(), 0, 0)

  function MPO(N::Int,
               A::Vector{<:ITensor},
               llim::Int = 0,
               rlim::Int = N+1)
    new(N, A, llim, rlim)
  end

  function MPO(sites::Vector{<:Index})
    N = length(sites)
    v = Vector{ITensor}(undef, N)
    l = [Index(1, "Link,l=$ii") for ii ∈ 1:N-1]
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
    new(N,v,0,N+1)
  end
 
end

MPO(A::Vector{<:ITensor}) = MPO(length(A),A,0,length(A)+1)

MPO(N::Int) = MPO(N,fill(ITensor(),N))

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
  MPO(N,its)
end

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
"""
siteinds(A::MPO, x::MPS) = [siteind(A, x, j) for j in eachindex(A)]

"""
dot(y::MPS, A::MPO, x::MPS)
inner(y::MPS, A::MPO, x::MPS)

Compute <y|A|x>
"""
function LinearAlgebra.dot(y::MPS,
                           A::MPO,
                           x::MPS)::Number
  N = length(A)
  if length(y) != N || length(x) != N
      throw(DimensionMismatch("inner: mismatched lengths $N and $(length(x)) or $(length(y))"))
  end
  ydag = dag(y)
  simlinkinds!(ydag)
  sAx = siteinds(A, x)
  replacesiteinds!(ydag, sAx)
  O = ydag[1]*A[1]*x[1]
  for j in 2:N
    O = O*ydag[j]*A[j]*x[j]
  end
  return O[]
end

"""
dot(B::MPO, y::MPS, A::MPO, x::MPS)
inner(B::MPO, y::MPS, A::MPO, x::MPS)

Compute <By|A|x>
"""
function LinearAlgebra.dot(B::MPO,
                           y::MPS,
                           A::MPO,
                           x::MPS)::Number
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

"""
error_mpoprod(y::MPS, A::MPO, x::MPS)

Compute the distance between A|x> and an approximation MPS y:
| |y> - A|x> |/| A|x> | = √(1 + (<y|y> - 2*real(<y|A|x>))/<Ax|A|x>)
"""
function error_mpoprod(y::MPS, A::MPO, x::MPS)
  N = length(A)
  if length(y) != N || length(x) != N
    throw(DimensionMismatch("inner: mismatched lengths $N and $(length(x)) or $(length(y))"))
  end
  iyy = inner(y, y)
  iyax = inner(y, A, x)
  iaxax = inner(A, x, A, x) 
  return sqrt(abs(1. + (iyy - 2*real(iyax))/iaxax))
end

function applympo(A::MPO, psi::MPS; kwargs...)::MPS
  method = get(kwargs, :method, "densitymatrix")
  if method == "DensityMatrix"
    @warn "In applympo, method DensityMatrix is deprecated in favor of densitymatrix"
    method = "densitymatrix"
  end
  if method == "densitymatrix"
    return applympo_densitymatrix(A, psi; kwargs...)
  elseif method == "naive" || method == "Naive"
    return applympo_naive(A, psi; kwargs...)
  end
  throw(ArgumentError("Method $method not supported"))
end

function applympo_densitymatrix(A::MPO, psi::MPS; kwargs...)::MPS
  n = length(A)
  n != length(psi) && throw(DimensionMismatch("lengths of MPO ($n) and MPS ($(length(psi))) do not match"))
  psi_out         = similar(psi)
  cutoff::Float64 = get(kwargs, :cutoff, 1e-13)

  maxdim::Int     = get(kwargs,:maxdim,maxlinkdim(psi))
  mindim::Int     = max(get(kwargs,:mindim,1), 1)
  normalize::Bool = get(kwargs, :normalize, false) 
  all(x -> x != Index(),
      [siteind(A, psi, j) for j in 1:n]) || 
  throw(ErrorException("MPS and MPO have different site indices in applympo method 'DensityMatrix'"))

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
  psi_out[n] = setprime(dag(FU), 0, "Site")
  O     = O * FU * psi[n-1] * A[n-1]
  O     = noprime(O, "Site")
  for j in reverse(2:n-1)
    dO  = prime(dag(O), rand_plev)
    ρ   = E[j-1] * O * dO
    ts  = tags(commonind(psi[j], psi[j-1]))
    Lis = IndexSet(commonind(ρ, A[j]), commonind(ρ, psi_out[j+1])) 
    Ris = prime(Lis, rand_plev)
    FU, D = eigen(ρ, Lis, Ris; ishermitian=true,
                               tags=ts, 
                               kwargs...)
    psi_out[j] = dag(FU)
    O = O * FU * psi[j-1] * A[j-1]
    O = noprime(O, "Site")
  end
  if normalize
    O /= norm(O)
  end
  psi_out[1]    = copy(O)
  setleftlim!(psi_out, 0)
  setrightlim!(psi_out, 2)
  return psi_out
end

function applympo_naive(A::MPO, psi::MPS; kwargs...)::MPS
  truncate = get(kwargs,:truncate,true)

  N = length(A)
  if N != length(psi) 
    throw(DimensionMismatch("lengths of MPO ($N) and MPS ($(length(psi))) do not match"))
  end

  psi_out = MPS(N)
  for j=1:N
    psi_out[j] = noprime(A[j]*psi[j])
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

function multmpo(A::MPO, B::MPO; kwargs...)::MPO
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

    for i in 1:N
        if length(intersect(inds(A_[i], "Site"), inds(B_[i], "Site"))) == 2
            A_[i] = prime(A_[i], "Site")
        end
    end
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
    @inbounds for (AA, BB) in zip(data(A_), data(B_))
        sda = setdiff(inds(AA, "Site"), inds(BB, "Site"))
        sdb = setdiff(inds(BB, "Site"), inds(AA, "Site"))
        sda_ind = setprime(sda[1], 0) == sdb[1] ? plev(sda[1]) == 1 ? sda[1] : setprime(sda[1], 1) : setprime(sda[1], 0)
        push!(sites_A, sda_ind)
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
        res[i], nfork = factorize(mapprime(clust,2,1),
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
    for i in 1:N
        res[i] = mapprime(res[i], 2, 1)
    end
    return res
end

