using KrylovKit: Lanczos, eigsolve

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
      @show scalar(phi*phi)
      Hphi = PH(phi)
      @show inds(Hphi)
      @show scalar(phi*Hphi)

      lczos = Lanczos(krylovdim=2,maxiter=2,tol=10*eps(Float64))
      vals, vecs, info = eigsolve(PH,phi,1,:SR,lczos)
      @show vals
      exit(0)

      #dir = ha==1 ? "Fromleft" : "Fromright"
      #replaceBond!(psi,b,phi,dir;kwargs...)
    end
  end
  return (energy,psi)
end

