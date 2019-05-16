mutable struct ProjMPO
  lpos::Int
  rpos::Int
  nsite::Int
  H::MPO
  LR::Vector{ITensor}
  ProjMPO(H::MPO) = new(0,length(H)+1,2,H,fill(ITensor(),length(H)))
end

nsite(ph::ProjMPO) = ph.nsite
length(ph::ProjMPO) = length(ph.H)

function LProj(ph::ProjMPO)::ITensor
  (ph.lpos <= 0) && return ITensor()
  return ph.LR[ph.lpos]
end
function RProj(ph::ProjMPO)::ITensor
  (ph.rpos >= length(ph)+1) && return ITensor()
  return ph.LR[ph.rpos]
end

function makeL!(ph::ProjMPO,
                psi::MPS,
                k::Int)
  while ph.lpos < k
    ll = ph.lpos
    if ll <= 0
      ph.LR[1] = psi[1]*ph.H[1]*dag(prime(psi[1]))
      ph.lpos = 1
    else
      ph.LR[ll+1] = ph.LR[ll]*psi[ll+1]*ph.H[ll+1]*dag(prime(psi[ll+1]))
      ph.lpos += 1
    end
  end
end

function makeR!(ph::ProjMPO,
                psi::MPS,
                k::Int)
  N = length(ph.H)
  while ph.rpos > k
    rl = ph.rpos
    if rl >= N+1
      ph.LR[N] = psi[N]*ph.H[N]*dag(prime(psi[N]))
      ph.rpos = N
    else
      ph.LR[rl-1] = ph.LR[rl]*psi[rl-1]*ph.H[rl-1]*dag(prime(psi[rl-1]))
      ph.rpos -= 1
    end
  end
end

function position!(ph::ProjMPO,
                   psi::MPS, 
                   pos::Int)
  makeL!(ph,psi,pos-1)
  makeR!(ph,psi,pos+nsite(ph))

  #These next two lines are needed 
  #when moving LProj and RProj backward
  ph.lpos = pos-1
  ph.rpos = pos+nsite(ph)
end

