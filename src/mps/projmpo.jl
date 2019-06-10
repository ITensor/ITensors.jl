mutable struct ProjMPO
  lpos::Int
  rpos::Int
  nsite::Int
  H::MPO
  LR::Vector{ITensor}
  ProjMPO(H::MPO) = new(0,length(H)+1,2,H,fill(ITensor(),length(H)))
end

nsite(pm::ProjMPO) = pm.nsite
length(pm::ProjMPO) = length(pm.H)


function LProj(pm::ProjMPO)::ITensor
  (pm.lpos <= 0) && return ITensor()
  return pm.LR[pm.lpos]
end

function RProj(pm::ProjMPO)::ITensor
  (pm.rpos >= length(pm)+1) && return ITensor()
  return pm.LR[pm.rpos]
end

L_t = 0.0
C_t = 0.0
R_t = 0.0

function product(pm::ProjMPO,
                 v::ITensor)::ITensor
  Hv = v
  if isNull(LProj(pm))
    if !isNull(RProj(pm))
      Hv *= RProj(pm)
    end
    for j=pm.rpos-1:-1:pm.lpos+1
      Hv *= pm.H[j]
    end
  else #if LProj is not null
    Hv *= LProj(pm)
    for j=pm.lpos+1:pm.rpos-1
      Hv *= pm.H[j]
    end
    if !isNull(RProj(pm))
      Hv *= RProj(pm)
    end
  end
  return noprime(Hv)
end

(pm::ProjMPO)(v::ITensor) = product(pm,v)

function size(pm::ProjMPO)::Tuple{Int,Int}
  d = 1
  if !isNull(LProj(pm))
    for i in inds(LProj(pm))
      plev(i) > 0 && (d *= dim(i))
    end
  end
  for j=pm.lpos+1:pm.rpos-1
    for i in inds(pm.H[j])
      plev(i) > 0 && (d *= dim(i))
    end
  end
  if !isNull(RProj(pm))
    for i in inds(RProj(pm))
      plev(i) > 0 && (d *= dim(i))
    end
  end
  return (d,d)
end

function makeL!(pm::ProjMPO,
                psi::MPS,
                k::Int)
  while pm.lpos < k
    ll = pm.lpos
    if ll <= 0
      pm.LR[1] = psi[1]*pm.H[1]*dag(prime(psi[1]))
      pm.lpos = 1
    else
      pm.LR[ll+1] = pm.LR[ll]*psi[ll+1]*pm.H[ll+1]*dag(prime(psi[ll+1]))
      pm.lpos += 1
    end
  end
end

function makeR!(pm::ProjMPO,
                psi::MPS,
                k::Int)
  N = length(pm.H)
  while pm.rpos > k
    rl = pm.rpos
    if rl >= N+1
      pm.LR[N] = psi[N]*pm.H[N]*dag(prime(psi[N]))
      pm.rpos = N
    else
      pm.LR[rl-1] = pm.LR[rl]*psi[rl-1]*pm.H[rl-1]*dag(prime(psi[rl-1]))
      pm.rpos -= 1
    end
  end
end

function position!(pm::ProjMPO,
                   psi::MPS, 
                   pos::Int)
  makeL!(pm,psi,pos-1)
  makeR!(pm,psi,pos+nsite(pm))

  #These next two lines are needed 
  #when moving LProj and RProj backward
  pm.lpos = pos-1
  pm.rpos = pos+nsite(pm)
end

