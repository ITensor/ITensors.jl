export MPS,
       position!,
       prime!,
       primelinks!,
       simlinks!,
       inner,
       productMPS,
       randomMPS,
       maxLinkDim,
       linkindex,
       siteindex,
       siteinds


mutable struct MPS
  N_::Int
  A_::Vector{ITensor}
  llim_::Int
  rlim_::Int

  MPS() = new(0,Vector{ITensor}(),0,0)

  MPS(N::Int) = MPS(N,Vector{ITensor}(undef,N),0,N+1)

  function MPS(N::Int, 
               A::Vector{ITensor}, 
               llim::Int=0, 
               rlim::Int=N+1)
    new(N,A,llim,rlim)
  end
end

function MPS(sites::SiteSet)
  N = length(sites)
  v = Vector{ITensor}(undef, N)
  l = [Index(1, "Link,l=$ii") for ii=1:N-1]
  for ii in eachindex(sites)
    s = sites[ii]
    if ii == 1
      v[ii] = ITensor(l[ii], s)
    elseif ii == N
      v[ii] = ITensor(l[ii-1], s)
    else
      v[ii] = ITensor(l[ii-1],s,l[ii])
    end
  end
  return MPS(N,v,0,N+1)
end

length(m::MPS) = m.N_
tensors(m::MPS) = m.A_
leftLim(m::MPS) = m.llim_
rightLim(m::MPS) = m.rlim_

getindex(m::MPS, n::Integer) = getindex(tensors(m),n)
setindex!(m::MPS,T::ITensor,n::Integer) = setindex!(tensors(m),T,n)

copy(m::MPS) = MPS(m.N_,copy(tensors(m)),m.llim_,m.rlim_)
similar(m::MPS) = MPS(m.N_, similar(tensors(m)), 0, m.N_)

eachindex(m::MPS) = 1:length(m)

function show(io::IO, M::MPS)
  print(io,"MPS")
  (length(M) > 0) && print(io,"\n")
  for (i, A) ∈ enumerate(tensors(M))
    println(io,"$i  $(inds(A))")
  end
end

function randomMPS(sites)
  M = MPS(sites)
  for i in eachindex(sites)
    randn!(M[i])
    normalize!(M[i])
  end
  M.llim_ = 1
  M.rlim_ = length(M)
  return M
end

const InitState = Union{Vector{String},Vector{Int}}

function productMPS(sites::SiteSet,
                    is::InitState)
  N = length(sites)
  if N != length(is)
    throw(DimensionMismatch("Site Set and InitState sizes don't match"))
  end
  As = Vector{ITensor}(undef,N)
  links  = Vector{Index}(undef,N)
  for n in eachindex(is)
    s = sites[n]
    links[n] = Index(1,"Link,l=$n")
    if n == 1
      A = ITensor(s,links[n])
      A[state(sites,n,is[n]),links[n](1)] = 1.0
    elseif n == N
      A = ITensor(links[n-1],s)
      A[links[n-1](1),state(sites,n,is[n])] = 1.0
    else
      A = ITensor(links[n-1],s,links[n])
      A[links[n-1](1),state(sites,n,is[n]),links[n](1)] = 1.0
    end
    As[n] = A
  end
  return MPS(N,As,0,2)
end

function linkindex(M::MPS,j::Integer) 
  N = length(M)
  j ≥ length(M) && error("No link index to the right of site $j (length of MPS is $N)")
  li = commonindex(M[j],M[j+1])
  if isdefault(li)
    error("linkindex: no MPS link index at link $j")
  end
  return li
end

function siteindex(M::MPS,j::Integer)
  N = length(M)
  if j == 1
    si = uniqueindex(M[j],M[j+1])
  elseif j == N
    si = uniqueindex(M[j],M[j-1])
  else
    si = uniqueindex(M[j],(M[j-1],M[j+1]))
  end
  return si
end

function siteinds(M::MPS)
  N = length(M)
  is = IndexSet(N)
  for j in eachindex(M)
    is[j] = siteindex(M,j)
  end
  return is
end

function replacesites!(M::MPS,sites)
  for j in eachindex(M)
    sj = siteindex(M,j)
    replaceindex!(M[j],sj,sites[j])
  end
  return M
end

function position!(M::MPS,
                   j::Integer)
  N = length(M)
  while leftLim(M) < (j-1)
    ll = leftLim(M)+1
    s = findindex(M[ll],"Site")
    if ll == 1
      (Q,R) = qr(M[ll],s)
    else
      li = linkindex(M,ll-1)
      (Q,R) = qr(M[ll],s,li)
    end
    M[ll] = Q
    M[ll+1] *= R
    M.llim_ += 1
  end

  while rightLim(M) > (j+1)
    rl = rightLim(M)-1
    s = findindex(M[rl],"Site")
    if rl == N
      (Q,R) = qr(M[rl],s)
    else
      ri = linkindex(M,rl)
      (Q,R) = qr(M[rl],s,ri)
    end
    M[rl] = Q
    M[rl-1] *= R
    M.rlim_ -= 1
  end
  M.llim_ = j-1
  M.rlim_ = j+1
end

"""
inner(psi::MPS, phi::MPS)

Compute <psi|phi>
"""
function inner(M1::MPS, M2::MPS)::Number
  N = length(M1)
  if length(M2) != N
      throw(DimensionMismatch("inner: mismatched lengths $N and $(length(M2))"))
  end
  M1dag = dag(M1)
  simlinks!(M1dag)
  O = M1dag[1]*M2[1]
  @inbounds for j ∈ eachindex(M1)[2:end]
    O *= M1dag[j]*M2[j]
  end
  return O[]
end

function replaceBond!(M::MPS,
                      b::Int,
                      phi::ITensor;
                      kwargs...)
  FU,FV = factorize(phi,inds(M[b]); which_factorization="automatic", kwargs...)
  M[b]   = FU
  M[b+1] = FV
end

