mutable struct LocalMPO
  lpos::Int
  rpos::Int
  H::MPO
  LR::Vector{ITensor}
  LocalMPO(H::MPO) = new(1,length(H)+1,H,Vector{ITensor}(undef,length(H)))
end

L(lm::LocalMPO) = lm.LR[lm.lpos]
R(lm::LocalMPO) = lm.LR[lm.rpos]

function position!(lm::LocalMPO,
                   psi::MPS, 
                   pos::Int)
  while lm.lpos < (pos-1)
    ll = lm.lpos
    if ll == 1
      lm.LR[1] = psi(1)*lm.H[1]*dag(prime(psi(1)))
    else
      lm.LR[ll+1] = psi(ll)*lm.H[ll]*dag(prime(psi(ll)))
    end
    lm.lpos += 1
  end

  N = length(lm.H)
  while lm.rpos > (pos+1)
    rl = lm.rpos
    if rr == N
      lm.LR[N] = psi(N)*lm.H[N]*dag(prime(psi(N)))
    else
      lm.LR[rl-1] = psi(rl)*lm.H[rl]*dag(prime(psi(rl)))
    end
    lm.rpos -= 1
  end
end

function dmrg!(psi::MPS,
               H::MPO,
               sweeps::Sweeps;
               kwargs...)

  N = length(psi)

  PH = LocalMPO(H)
  
  position!(PH,psi,1)

  for sw=1:nsweep(sweeps)
    for (b,ha) in sweepnext(N)
      @printf "sw=%d ha=%d b=%d\n" sw ha b

      position!(PH,psi,b)

      phi = psi[b]*psi[b+1]

      #phi,energy = davidson(PH,phi;kwargs...)

      #dir = ha==1 ? "Fromleft" : "Fromright"
      #replaceBond!(psi,b,phi,dir;kwargs...)
    end
  end
end

