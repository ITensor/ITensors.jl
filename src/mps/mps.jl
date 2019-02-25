
mutable struct MPS
  N_::Int
  A_::Vector{ITensor}
  llim_::Int
  rlim_::Int

  MPS() = new(0,Vector{ITensor}(),0,0)

  function MPS(N::Int, A::Vector{ITensor}, llim::Int, rlim::Int)
    new(N,A,llim,rlim)
  end
  

  function MPS(sites::SiteSet) # random MPS
    N = length(sites)
    new(N,fill(ITensor(),N),0,N+1)
  end
  function MPS(is::InitState; store_type::DataType=Float64)
    N = length(is)
    its = Vector{ITensor}(undef, length(is))
    spin_sites = Vector{Site}(undef, length(is))
    link_inds  = Vector{Index}(undef, length(is))
    for ii in 1:N
        i_is = is[ii]
        i_site = site(is, ii)
        spin_sites[ii] = i_site.dim == 2 ? SpinSite{Val{1//2}}(i_site) : SpinSite{Val{1}}(i_site)
        spin_op = op(store_type, spin_sites[ii], i_is)
        link_inds[ii] = Index(1, "Link,n=$ii")
        s = i_site 
        local this_it
        if ii == 1
            this_it = ITensor(store_type, link_inds[ii], i_site)
            this_it[link_inds[ii](1), s[:]] = spin_op[s[:]]
        elseif ii == N
            this_it = ITensor(store_type, link_inds[ii-1], i_site)
            this_it[link_inds[ii-1](1), s[:]] = spin_op[s[:]]
        else
            this_it = ITensor(store_type, link_inds[ii-1], link_inds[ii], i_site)
            this_it[link_inds[ii-1](1), link_inds[ii](1), s[:]] = spin_op[s[:]]
        end
        its[ii] = this_it
    end
    # construct InitState from SiteSet -- is(sites, "Up")
    new(N,its,0,2)
  end
end
MPS(N::Int, d::Int, opcode::String; store_type::DataType=Float64) = MPS(InitState(Sites(N,d), opcode), store_type=store_type)
MPS(s::SiteSet, opcode::String; store_type::DataType=Float64) = MPS(InitState(s, opcode), store_type=store_type)

length(m::MPS) = m.N_
leftLim(m::MPS) = m.llim_
rightLim(m::MPS) = m.rlim_

getindex(m::MPS, n::Integer) = getindex(m.A_,n)
setindex!(m::MPS,T::ITensor,n::Integer) = setindex!(m.A_,T,n)

copy(m::MPS) = MPS(m.N_,copy(m.A_),m.llim_,m.rlim_)

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

function overlap(psi1::MPS,
                 psi2::MPS)::Number64
  N = length(psi1)
  if length(psi2) != N
    error("overlap: mismatched lengths $N and $(length(psi2))")
  end

  s1 = findtags(psi2[1],"Site")
  O = psi1[1]*primeexcept(psi2[1],s1)
  for j=2:N
    sj = findtags(psi2[j],"Site")
    O *= psi1[j]
    O *= primeexcept(psi2[j],sj)
  end
  return O[]
end

function randomMPS(sites::SiteSet,
                   m::Int=1)
  psi = MPS(sites)
  for i=1:length(psi)
    psi[i] = randomITensor(sites[i])
    psi[i] /= norm(psi[i])
  end
  if m > 1
    error("randomMPS: currently only m==1 supported")
  end
  return psi
end
