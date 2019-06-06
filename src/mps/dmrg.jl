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

    phi_t = 0.0
    eigen_t = 0.0
    svd_t = 0.0
    sweep_t = 0.0
    global prod_t = 0.0
    global L_t = 0.0
    global C_t = 0.0
    global R_t = 0.0
    global small_eigen_t = 0.0
    global orth_t = 0.0
    global contract_t = 0.0
    global gemm_t = 0.0
    global permute_t = 0.0

    sweep_t += @elapsed begin
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
    end
    @printf "After sweep %d energy=%.12f maxDim=%d\n" sw energy maxDim(psi)
    @show phi_t
    @show eigen_t
    @printf "  prod_t = %.12f\n" prod_t
    @printf "    L_t = %.12f\n" L_t
    @printf "    C_t = %.12f\n" C_t
    @printf "    R_t = %.12f\n" R_t
    @printf "    Total  = %.12f (?= %.12f)\n" L_t+C_t+R_t prod_t
    @printf "  small_eigen_t = %.12f\n" small_eigen_t
    @printf "  orth_t = %.12f\n" orth_t
    @printf "  Total  = %.12f (?= %.12f)\n" prod_t+small_eigen_t+orth_t eigen_t
    @printf "contract_t = %.12f\n" contract_t
    @printf "  gemm_t = %.12f\n" gemm_t
    @printf "  permute_t = %.12f\n" permute_t
    @printf "  Total  = %.12f (?= %.12f)\n" gemm_t+permute_t contract_t
    @show svd_t
    @printf "sweep_t = %.12f (phi_t+eigen_t+svd_t = %.12f)\n" sweep_t phi_t+eigen_t+svd_t
    println()
    println()
  end
  return (energy,psi)
end

