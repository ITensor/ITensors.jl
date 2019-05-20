
mutable struct MPS
  N_::Int
  A_::Vector{ITensor}
  llim_::Int
  rlim_::Int

  MPS() = new(0,Vector{ITensor}(),0,0)

  function MPS(N::Int, A::Vector{ITensor}, llim::Int, rlim::Int)
    new(N,A,llim,rlim)
  end
  
  function MPS(sites::SiteSet)
    N = length(sites)
    v = Vector{ITensor}(undef, N)
    l = [Index(1, "Link,l=$ii") for ii ∈ 1:N-1]
    for ii in 1:N
      s = sites[ii]
      if ii == 1
        v[ii] = ITensor(l[ii], s)
      elseif ii == N
        v[ii] = ITensor(l[ii-1], s)
      else
        v[ii] = ITensor(l[ii-1], l[ii], s)
      end
    end
    new(N,v,0,N+1)
  end

  function MPS(::Type{T}, is::InitState) where {T}
    N = length(is)
    its = Vector{ITensor}(undef, length(is))
    link_inds  = Vector{Index}(undef, length(is))
    for ii in 1:N
        i_is = is[ii]
        i_site = site(is, ii)
        spin_op = op(T(i_site), i_is)
        link_inds[ii] = Index(1, "Link,n=$ii")
        s = i_site 
        local this_it
        if ii == 1
            this_it = ITensor(link_inds[ii], i_site)
            this_it[link_inds[ii](1), s[:]] = spin_op[s[:]]
        elseif ii == N
            this_it = ITensor(link_inds[ii-1], i_site)
            this_it[link_inds[ii-1](1), s[:]] = spin_op[s[:]]
        else
            this_it = ITensor(link_inds[ii-1], link_inds[ii], i_site)
            this_it[link_inds[ii-1](1), link_inds[ii](1), s[:]] = spin_op[s[:]]
        end
        its[ii] = this_it
    end
    # construct InitState from SiteSet -- is(sites, "Up")
    new(N,its,0,2)
  end
end
MPS(N::Int, d::Int, opcode::String) = MPS(InitState(Sites(N,d), opcode))
MPS(N::Int) = MPS(N,Vector{ITensor}(undef,N),0,N+1)
MPS(s::SiteSet, opcode::String) = MPS(InitState(s, opcode))

length(m::MPS) = m.N_
leftLim(m::MPS) = m.llim_
rightLim(m::MPS) = m.rlim_

getindex(m::MPS, n::Integer) = getindex(m.A_,n)
setindex!(m::MPS,T::ITensor,n::Integer) = setindex!(m.A_,T,n)

copy(m::MPS) = MPS(m.N_,copy(m.A_),m.llim_,m.rlim_)

function dag(m::MPS)
  N = length(m)
  mdag = MPS(N)
  for i ∈ 1:N
    mdag[i] = dag(m[i])
  end
  return mdag
end

function show(io::IO,
              psi::MPS)
  print(io,"MPS")
  (length(psi) > 0) && print(io,"\n")
  for i=1:length(psi)
    println(io,"$i  $(psi[i])")
  end
end

function linkind(psi::MPS,j::Integer) 
  li = commonindex(psi[j],psi[j+1])
  if isdefault(li)
    error("linkind: no MPS link index at link $j")
  end
  return li
end

function simlinks!(m::MPS)
  N = length(m)
  for i ∈ 1:N-1
    l = linkind(m,i)
    l̃ = sim(l)
    m[i] *= δ(l,l̃)
    m[i+1] *= δ(l,l̃)
  end
end

function position!(psi::MPS,
                   j::Integer)
  N = length(psi)

  while leftLim(psi) < (j-1)
    ll = leftLim(psi)+1
    s = findtags(psi[ll],"Site")
    if ll == 1
      (Q,R) = qr(psi[ll],s)
    else
      li = linkind(psi,ll-1)
      (Q,R) = qr(psi[ll],s,li)
    end
    psi[ll] = Q
    psi[ll+1] *= R
    psi.llim_ += 1
  end

  while rightLim(psi) > (j+1)
    rl = rightLim(psi)-1
    s = findtags(psi[rl],"Site")
    if rl == N
      (Q,R) = qr(psi[rl],s)
    else
      ri = linkind(psi,rl)
      (Q,R) = qr(psi[rl],s,ri)
    end
    psi[rl] = Q
    psi[rl-1] *= R
    psi.rlim_ -= 1
  end
end

function inner(psi1::MPS,
               psi2::MPS)::Number
  N = length(psi1)
  if length(psi2) != N
    error("inner: mismatched lengths $N and $(length(psi2))")
  end
  psi1dag = dag(psi1)
  simlinks!(psi1dag)
  O = psi1dag[1]*psi2[1]
  for j=2:N
    O *= psi1dag[j]*psi2[j]
  end
  return O[]
end

function randomMPS(sites::SiteSet,
                   m::Int=1)
  psi = MPS(sites)
  for i=1:length(psi)
    randn!(psi[i])
    psi[i] /= norm(psi[i])
  end
  if m > 1
    error("randomMPS: currently only m==1 supported")
  end
  return psi
end

function replaceBond!(psi::MPS,
                      b::Int,
                      phi::ITensor,
                      dir::String;
                      kwargs...)
  U,S,V,u,v = svd(phi,inds(psi[b]);kwargs...)
  if dir=="Fromleft"
    psi[b]   = U
    psi[b+1] = S*V
  elseif dir=="Fromright"
    psi[b]   = U*S
    psi[b+1] = V
  end
end

function maxDim(psi::MPS)
  md = 1
  for b=1:length(psi)-1
    md = max(md,dim(linkind(psi,b)))
  end
  return md
end
