export dmrg

function dmrg(H::MPO,
              psi0::MPS,
              sweeps::Sweeps;
              kwargs...)::Tuple{Float64,MPS}

  which_factorization::String = get(kwargs,:which_factorization,"automatic")

  psi = copy(psi0)
  N = length(psi)

  PH = ProjMPO(H)
  position!(PH,psi0,1)
  energy = 0.0

  for sw=1:nsweep(sweeps)
    sw_time = @elapsed begin

    @debug begin
      reset!(timer)
      println("\n\n~~~ This is sweep $sw ~~~\n\n")
    end

    for (b,ha) in sweepnext(N)
      position!(PH,psi,b)

      phi = psi[b]*psi[b+1]

      energy,phi = davidson(PH,phi;kwargs...)

      dir = ha==1 ? "fromleft" : "fromright"
      replaceBond!(psi,b,phi;
                   maxdim=maxdim(sweeps,sw),
                   mindim=mindim(sweeps,sw),
                   cutoff=cutoff(sweeps,sw),
                   dir=dir,
                   which_factorization=which_factorization)
    end
    end
    @printf "After sweep %d energy=%.12f maxDim=%d time=%.3f\n" sw energy maxDim(psi) sw_time
    @debug printTimes(timer)
  end
  return (energy,psi)
end

@doc """ 
Use the density matrix renormalization group (DMRG) algorithm to
optimize a matrix product state (MPS) to be the eigenvector
of the Hermitian matrix product operator (MPO) H with minimal 
eigenvalue.

Inputs:
* `H::MPO` - a Hermitian MPO
* `psi0::MPS` - MPS used to initialize the optimization
* `sweeps::Sweeps` - `Sweeps` object used to control the algorithm

Returns:
* `energy::Float64` - eigenvalue of the optimized MPS
* `psi::MPS` - optimized MPS

""" dmrg
