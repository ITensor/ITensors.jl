
function dmrg(H::MPO,
              psi0::MPS,
              sweeps::Sweeps;
              kwargs...)::Tuple{Float64,MPS}
  psi = copy(psi0)
  N = length(psi)

  PH = ProjMPO(H)
  position!(PH,psi0,1)

  energy = 0.0

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
  return (energy,psi)
end

