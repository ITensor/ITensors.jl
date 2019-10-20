export dmrg


function dmrg(H::MPO,
              psi0::MPS,
              sweeps::Sweeps;
              kwargs...)::Tuple{Float64,MPS}

  which_factorization::String = get(kwargs,:which_factorization,"automatic")
  obs = get(kwargs,:observer, NoObserver())
  quiet::Bool = get(kwargs,:quiet,false)

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

@timeit_debug GLOBAL_TIMER "davidson" begin
      energy,phi = davidson(PH,phi;kwargs...)
end

      dir = ha==1 ? "fromleft" : "fromright"

@timeit_debug GLOBAL_TIMER "replaceBond!" begin
      replaceBond!(psi,b,phi;
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
                   quiet=quiet)
    end
    end
    if !quiet
      @printf("After sweep %d energy=%.12f maxLinkDim=%d time=%.3f\n",sw,energy,maxLinkDim(psi),sw_time)
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
