export dmrg


function dmrg(H::MPO,
              psi0::MPS,
              sweeps::Sweeps;
              kwargs...)::Tuple{Float64,MPS}
  which_factorization::String = get(kwargs,:which_factorization,"automatic")
  obs = get(kwargs,:observer, NoObserver())
  quiet::Bool = get(kwargs,:quiet,false)

  # eigsolve kwargs
  eigsolve_tol::Float64 = get(kwargs,:eigsolve_tol,1E-14)
  eigsolve_krylovdim::Int = get(kwargs,:eigsolve_krylovdim,3)
  eigsolve_maxiter::Int = get(kwargs,:eigsolve_maxiter,1)
  eigsolve_verbosity::Int = get(kwargs,:eigsolve_verbosity,0)

  # TODO: add support for non-Hermitian DMRG
  ishermitian::Bool = true #get(kwargs,:ishermitian,true)

  # TODO: add support for targeting other states with DMRG
  # (such as the state with the largest eigenvalue)
  eigsolve_which_eigenvalue::Symbol = :SR #get(kwargs,:eigsolve_which_eigenvalue,:SR)

  # Keyword argument deprecations
  haskey(kwargs,:maxiter) && error("""maxiter keyword has been replace by eigsolve_krylovdim.
                                      Note: compared to the C++ version of ITensor,
                                      setting eigsolve_krylovdim 3 is the same as setting
                                      a maxiter of 2.""")
  haskey(kwargs,:errgoal) && error("errgoal keyword has been replace by eigsolve_tol.")

  psi = copy(psi0)
  N = length(psi)

  PH = ProjMPO(H)
  position!(PH,psi0,1)
  energy = 0.0

  for sw=1:nsweep(sweeps)
    sw_time = @elapsed begin

    for (b,ha) in sweepnext(N)

@timeit_debug GLOBAL_TIMER "position!" begin
      position!(PH,psi,b)
end

@timeit_debug GLOBAL_TIMER "psi[b]*psi[b+1]" begin
      phi = psi[b]*psi[b+1]
end

@timeit_debug GLOBAL_TIMER "eigsolve" begin
      vals,vecs = eigsolve(PH,phi,1,eigsolve_which_eigenvalue; ishermitian=ishermitian,
                                                               tol=eigsolve_tol,
                                                               krylovdim=eigsolve_krylovdim,
                                                               maxiter=eigsolve_maxiter)
end
      energy,phi = vals[1],vecs[1]

      dir = ha==1 ? "fromleft" : "fromright"

@timeit_debug GLOBAL_TIMER "replacebond!" begin
      spec = replacebond!(psi,b,phi;
                          maxdim=maxdim(sweeps,sw),
                          mindim=mindim(sweeps,sw),
                          cutoff=cutoff(sweeps,sw),
                          dir=dir,
                          which_factorization=which_factorization)
end

      measure!(obs;energy=energy,
               psi=psi,
               bond=b,
               sweep=sw,
               half_sweep=ha,
               spec = spec,
               quiet=quiet)
    end
    end
    if !quiet
      @printf("After sweep %d energy=%.12f maxlinkdim=%d time=%.3f\n",sw,energy,maxlinkdim(psi),sw_time)
    end
    checkdone!(obs;quiet=quiet) && break
  end
  return (energy,psi)
end

@doc """
dmrg(H::MPO,psi0::MPS,sweeps::Sweeps;kwargs...)::Tuple{Float64,MPS}

Optimize a matrix product state (MPS) to be the eigenvector
of the Hermitian matrix product operator (MPO) H with minimal
eigenvalue using the density matrix renormalization group 
(DMRG) algorithm.

Inputs:
* `H::MPO` - a Hermitian MPO
* `psi0::MPS` - MPS used to initialize the optimization
* `sweeps::Sweeps` - `Sweeps` object used to control the algorithm

Returns:
* `energy::Float64` - eigenvalue of the optimized MPS
* `psi::MPS` - optimized MPS
""" dmrg
