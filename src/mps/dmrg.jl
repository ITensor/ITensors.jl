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

      noise_mag = noise(sweeps, sw)

      # This is slightly strange, but it does The Right Thing (TM).
      # replaceBond!() calls factorize(), which
      #   1. checks if noise > 0
      #   2. if so, ultimately calls _factorize_from_{left,right}_eigen;
      #   3. if not, may or may not call _eigen() (depending on other parameters.
      # But if noise == 0, _eigen() will ignore the noise tensor,
      # so it can be anything we want: here a zero-index default tensor.

      nt = (if noise_mag > 0 noisetensor(psi, PH, b, noise_mag, dir) else ITensor() end)


@timeit_debug GLOBAL_TIMER "replaceBond!" begin
      spec = replacebond!(psi,b,phi;
                          maxdim=maxdim(sweeps,sw),
                          mindim=mindim(sweeps,sw),
                          cutoff=cutoff(sweeps,sw),
                          noise=noise_mag,
                          noise_tensor=nt,
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
