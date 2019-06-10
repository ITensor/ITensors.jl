using KrylovKit: Lanczos, eigsolve


function iterEigSolve(PH,
                      phi::ITensor;
                      kwargs...)::Tuple{Number,ITensor}

  #
  # TODO: make version which takes krylovdim, maxiter
  #       was giving energies which randomly go up!
  #
  #tol = get(kwargs,:tol,10*eps(Float64))
  #krylovdim::Int = get(kwargs,:krylovdim,2)
  #maxiter::Int = get(kwargs,:maxiter,1)

  #actualdim = 1
  #for i in inds(phi)
  #  plev(i) == 0 && (actualdim *= dim(i))
  #end
  #if krylovdim > actualdim
  #  @printf "krylovdim=%d > actualdim=%d, resetting" krylovdim actualdim
  #  krylovdim = actualdim
  #end
  #lczos = Lanczos(tol=tol,krylovdim=krylovdim,maxiter=maxiter)
  #vals, vecs, info = eigsolve(PH,phi,1,:SR,lczos)

  vals, vecs, info = eigsolve(PH,phi,1,:SR,ishermitian=true)

  #@show info.normres[1]
  #@show info.numops
  #@show info.numiter
  return vals[1],vecs[1]
end

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
    sw_time = @elapsed begin

    sweep_t = 0.0
    phi_t = 0.0
    eigen_t = 0.0
    svd_t = 0.0

    reset!(timer)

    for (b,ha) in sweepnext(N)
      #@printf "sw=%d ha=%d b=%d\n" sw ha b

      position!(PH,psi,b)

      phi_t += @elapsed begin
        phi = psi[b]*psi[b+1]
      end
      #@printf "initial phi norm = %.3f\n" norm(phi)
      #@printf "initial energy = %.8f\n" scalar(phi*PH(phi))/norm(phi)^2

      eigen_t += @elapsed begin
        energy,phi = davidson(PH,phi;kwargs...)
      end

      #energy,phi = iterEigSolve(PH,phi;kwargs...)
      #@printf "unnorm energy=%.8f\n" scalar(phi*PH(phi))
      #phi /= norm(phi)
      #@printf "check phi energy = %.8f\n" scalar(phi*PH(phi))/norm(phi)^2

      svd_t += @elapsed begin
      dir = ha==1 ? "Fromleft" : "Fromright"
      replaceBond!(psi,b,phi,dir;
                   maxdim=maxdim(sweeps,sw),
                   mindim=mindim(sweeps,sw),
                   cutoff=cutoff(sweeps,sw))
      end

      #nphi = psi[b]*psi[b+1]
      #@printf "final MPS norm = %.3f\n" norm(nphi)
      #@printf "final energy = %.8f\n" scalar(nphi*PH(nphi))/norm(nphi)^2
      #@printf "dim=%d\n" dim(linkind(psi,b))

      #@printf "sw=%d ha=%d b=%d energy=%.8f dim=%d\n" sw ha b energy dim(linkind(psi,b))
      #pause()
    end
    @printf "After sweep %d energy=%.12f maxDim=%d time=%.3f\n" sw energy maxDim(psi) sw_time
    @show phi_t
    @show eigen_t
    @show svd_t
    @printf "sw_time = %.12f (phi_t+eigen_t+svd_t = %.12f)\n" sw_time phi_t+eigen_t+svd_t
    printTimes(timer)
    println("\n")
  end
  end
  return (energy,psi)
end

