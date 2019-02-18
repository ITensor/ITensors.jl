mutable struct LocalMPO
end

function dmrg!(psi::MPS,
               H::MPO,
               sweeps::Sweeps;
               kwargs...)

  N = length(psi)

  #PH = LocalMPO(H)
  
  #position!(psi,1)

  for sw=1:nsweep(sweeps)
    for (b,ha) in sweepnext(N)
      @printf "sw=%d ha=%d b=%d\n" sw ha b

      #position!(PH,b,psi)

      #phi = psi[b]*psi[b+1]

      #energy = davidson(PH,phi;kwargs...)

      #dir = ha==1 ? "Fromleft" : "Fromright"
      #replaceBond!(psi,b,phi,dir;kwargs...)
    end
  end
end

