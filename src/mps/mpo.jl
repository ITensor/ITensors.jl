export MPO,
       randomMPO,
       applyMPO 

struct MPO
  N_::Int
  A_::Vector{ITensor}

  MPO() = new(0,Vector{ITensor}())

  function MPO(N::Int, A::Vector{ITensor})
    new(N,A)
  end

  function MPO(sites::SiteSet)
    N = length(sites)
    v = Vector{ITensor}(undef, N)
    l = [Index(1, "Link,l=$ii") for ii ∈ 1:N-1]
    @inbounds for ii ∈ eachindex(sites)
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
    new(N,v)
  end
 
  function MPO(sites::SiteSet, 
               ops::Vector{String})
    N = length(sites)
    its = Vector{ITensor}(undef, N)
    links = Vector{Index}(undef, N)
    @inbounds for ii ∈ eachindex(sites)
        si = sites[ii]
        spin_op = op(sites, ops[ii], ii)
        links[ii] = Index(1, "Link,n=$ii")
        local this_it
        if ii == 1
            this_it = ITensor(links[ii], si, si')
            this_it[links[ii](1), si[:], si'[:]] = spin_op[si[:], si'[:]]
        elseif ii == N
            this_it = ITensor(links[ii-1], si, si')
            this_it[links[ii-1](1), si[:], si'[:]] = spin_op[si[:], si'[:]]
        else
            this_it = ITensor(links[ii-1], links[ii], si, si')
            this_it[links[ii-1](1), links[ii](1), si[:], si'[:]] = spin_op[si[:], si'[:]]
        end
        its[ii] = this_it
    end
    new(N,its)
  end

  function MPO(sites::SiteSet, 
               ops::String)
    return MPO(sites, fill(ops, length(sites)))
  end

end
MPO(N::Int) = MPO(N,Vector{ITensor}(undef,N))

function randomMPO(sites,
                   m::Int=1)
  M = MPO(sites)
  @inbounds for i ∈ eachindex(sites)
    randn!(M[i])
    normalize!(M[i])
  end
  m > 1 && throw(ArgumentError("randomMPO: currently only m==1 supported"))
  return M
end

length(m::MPO) = m.N_

getindex(m::MPO, n::Integer) = getindex(m.A_,n)
setindex!(m::MPO,T::ITensor,n::Integer) = setindex!(m.A_,T,n)

copy(m::MPO) = MPO(m.N_,copy(m.A_))

eachindex(m::MPO) = 1:length(m)

# TODO: optimize finding the index a little bit
# First do: scom = commonindex(A[j],x[j])
# Then do: uniqueindex(A[j],A[j-1],A[j+1],(scom,))
function siteindex(A::MPO,x::MPS,j::Integer)
  N = length(A)
  if j == 1
    si = uniqueindex(A[j],(A[j+1],x[j]))
  elseif j == N
    si = uniqueindex(A[j],(A[j-1],x[j]))
  else
    si = uniqueindex(A[j],(A[j-1],A[j+1],x[j]))
  end
  return si
end

function siteinds(A::MPO,x::MPS)
  is = IndexSet(length(A))
  @inbounds for j in eachindex(A)
    is[j] = siteindex(A,x,j)
  end
  return is
end

"""
    dag(m::MPS)
    dag(m::MPO)

Hermitian conjugation of a matrix product state or operator `m`.
"""

function dag(m::T) where {T <: Union{MPS, MPO}}
  N = length(m)
  mdag = T(N)
  @inbounds for i ∈ eachindex(m)
    mdag[i] = dag(m[i])
  end
  return mdag
end

function prime!(M::T,vargs...) where {T <: Union{MPS,MPO}}
  @inbounds for i ∈ eachindex(M)
    prime!(M[i],vargs...)
  end
end

function primelinks!(M::T, plinc::Integer = 1) where {T <: Union{MPS,MPO}}
  @inbounds for i ∈ eachindex(M)[1:end-1]
    l = linkindex(M,i)
    prime!(M[i],plinc,l)
    prime!(M[i+1],plinc,l)
  end
end

function simlinks!(M::T) where {T <: Union{MPS,MPO}}
  @inbounds for i ∈ eachindex(M)[1:end-1]
    l = linkindex(M,i)
    l̃ = sim(l)
    #M[i] *= δ(l,l̃)
    replaceindex!(M[i],l,l̃)
    #M[i+1] *= δ(l,l̃)
    replaceindex!(M[i+1],l,l̃)
  end
end

function maxDim(M::T) where {T <: Union{MPS,MPO}}
  md = 0
  for b ∈ eachindex(M)[1:end-1] 
    md = max(md,dim(linkindex(M,b)))
  end
  md
end

function show(io::IO, W::MPO)
  print(io,"MPO")
  (length(W) > 0) && print(io,"\n")
  @inbounds for (i, w) ∈ enumerate(inds.(W.A_))
    println(io,"$i  $w")
  end
end

function linkindex(M::MPO,j::Integer) 
  N = length(M)
  j ≥ length(M) && error("No link index to the right of site $j (length of MPO is $N)")
  li = commonindex(M[j],M[j+1])
  if isdefault(li)
    error("linkindex: no MPO link index at link $j")
  end
  return li
end

"""
inner(y::MPS, A::MPO, x::MPS)

Compute <y|A|x>
"""
function inner(y::MPS,
               A::MPO,
               x::MPS)::Number
  N = length(A)
  if length(y) != N || length(x) != N
      throw(DimensionMismatch("inner: mismatched lengths $N and $(length(x)) or $(length(y))"))
  end
  ydag = dag(y)
  simlinks!(ydag)
  sAx = siteinds(A,x)
  replacesites!(ydag,sAx)
  O = ydag[1]*A[1]*x[1]
  @inbounds for j ∈ eachindex(y)[2:end]
    O = O*ydag[j]*A[j]*x[j]
  end
  return O[]
end

function applyMPO(A::MPO, psi::MPS; kwargs...)::MPS
    method = get(kwargs, :method, "DensityMatrix")
    if method == "DensityMatrix"
        return densityMatrixApplyMPO(A, psi, kwargs...)
    end
    throw(ArgumentError("Method $method not supported"))
end

function densityMatrixApplyMPO(A::MPO, psi::MPS; kwargs...)::MPS
    n = length(A)
    n != length(psi) && throw(DimensionMismatch("lengths of MPO ($n) and MPS ($(length(psi))) do not match"))
    psi_out = similar(psi)
    cutoff::Float64 = get(kwargs, :cutoff, 1e-13)
    maxdim::Int = get(kwargs,:maxdim,maxDim(psi))
    mindim::Int = max(get(kwargs,:mindim,1), 1)
    normalize::Bool = get(kwargs, :normalize, false) 

    all(x->x!=Index(), [siteindex(A, psi, j) for j in 1:n]) || throw(ErrorException("MPS and MPO have different site indices in applyMPO method 'DensityMatrix'"))
    rand_plev = 14741

    psi_c = dag(copy(psi))
    prime!(psi_c, rand_plev)
    A_c = dag(copy(A))
    prime!(A_c, rand_plev)

    for j in 1:n-1
        unique_site_ind = setdiff(findinds(A_c[j], "Site"), findindex(psi_c[j], "Site"))[1]
        A_c[j] = setprime(A_c[j], 1, unique_site_ind)
    end
    E = Vector{ITensor}(undef, n-1)
    E[1] = psi[1]*A[1]*A_c[1]*psi_c[1]
    for j in 2:n-1
        E[j] = E[j-1]*psi[j]*A[j]*A_c[j]*psi_c[j]
    end
    O = psi[n] * A[n]
    ρ = E[n-1] * O * dag(prime(O, rand_plev))
    ts = tags(commonindex(psi[n], psi[n-1]))
    Lis = findinds(ρ, "n=$n,1")
    Ris = uniqueinds(ρ, Lis)
    FU, D = eigen(ρ, Lis, Ris, tags=ts)
    psi_out[n] = setprime(dag(FU), 0, "Site")
    O = O * FU * psi[n-1] * A[n-1]
    O = prime(O, -1, "Site")
    for j in reverse(2:n-1)
        dO = prime(dag(O), rand_plev)
        ρ = E[j-1] * O * dO
        ts = tags(commonindex(psi[j], psi[j-1]))
        Lis = findinds(ρ, "0")
        Ris = uniqueinds(ρ, Lis)
        FU, D = eigen(ρ, Lis, Ris, tags=ts)
        psi_out[j] = dag(FU)
        O = O * FU * psi[j-1] * A[j-1]
        O = prime(O, -1, "Site")
    end
    if normalize
        O /= norm(O)
    end
    psi_out[1] = O
    psi_out.llim_ = 0
    psi_out.rlim_ = 2
    return psi_out
end
