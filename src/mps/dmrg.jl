
"""
    dmrg(H::MPO,psi0::MPS,sweeps::Sweeps;kwargs...)
                    
Use the density matrix renormalization group (DMRG) algorithm
to optimize a matrix product state (MPS) such that it is the
eigenvector of lowest eigenvalue of a Hermitian matrix H,
represented as a matrix product operator (MPO).
The MPS `psi0` is used to initialize the MPS to be optimized,
and the `sweeps` object determines the parameters used to 
control the DMRG algorithm.

Returns:
* `energy::Float64` - eigenvalue of the optimized MPS
* `psi::MPS` - optimized MPS
"""
function dmrg(H::MPO,
              psi0::MPS,
              sweeps::Sweeps;
              kwargs...)
  PH = ProjMPO(H)
  return dmrg(PH,psi0,sweeps;kwargs...)
end

"""
    dmrg(Hs::Vector{MPO},psi0::MPS,sweeps::Sweeps;kwargs...)
                    
Use the density matrix renormalization group (DMRG) algorithm
to optimize a matrix product state (MPS) such that it is the
eigenvector of lowest eigenvalue of a Hermitian matrix H.
The MPS `psi0` is used to initialize the MPS to be optimized,
and the `sweeps` object determines the parameters used to 
control the DMRG algorithm.

This version of `dmrg` accepts a representation of H as a
Vector of MPOs, Hs = [H1,H2,H3,...] such that H is defined
as H = H1+H2+H3+...
Note that this sum of MPOs is not actually computed; rather
the set of MPOs [H1,H2,H3,..] is efficiently looped over at 
each step of the DMRG algorithm when optimizing the MPS.

Returns:
* `energy::Float64` - eigenvalue of the optimized MPS
* `psi::MPS` - optimized MPS
"""
function dmrg(Hs::Vector{MPO},
              psi0::MPS,
              sweeps::Sweeps;
              kwargs...)
  PHS = ProjMPOSum(Hs)
  return dmrg(PHS,psi0,sweeps;kwargs...)
end

"""
    dmrg(H::MPO,Ms::Vector{MPS},psi0::MPS,sweeps::Sweeps;kwargs...)
                    
Use the density matrix renormalization group (DMRG) algorithm
to optimize a matrix product state (MPS) such that it is the
eigenvector of lowest eigenvalue of a Hermitian matrix H,
subject to the constraint that the MPS is orthogonal to each
of the MPS provided in the Vector `Ms`. The orthogonality
constraint is approximately enforced by adding to H terms of 
the form w|M1><M1| + w|M2><M2| + ... where Ms=[M1,M2,...] and
w is the "weight" parameter, which can be adjusted through the
optional `weight` keyword argument.
The MPS `psi0` is used to initialize the MPS to be optimized,
and the `sweeps` object determines the parameters used to 
control the DMRG algorithm.

Returns:
* `energy::Float64` - eigenvalue of the optimized MPS
* `psi::MPS` - optimized MPS
"""
function dmrg(H::MPO,
              Ms::Vector{MPS},
              psi0::MPS,
              sweeps::Sweeps;
              kwargs...)
  weight = get(kwargs,:weight,1.0)
  PMM = ProjMPO_MPS(H,Ms;weight=weight)
  return dmrg(PMM,psi0,sweeps;kwargs...)
end

# Make the perturbation to the density matrix used in "noise term" DMRG
# This assumes that A comes in with no primes
# If it doesn't, I expect A² += drho later to fail
function deltarho(A :: ITensor, nt :: ITensor, is)
  drho = noprime(nt * A)
  drho *= prime(dag(drho), is)
end

function dmrg(PH,
              psi0::MPS,
              sweeps::Sweeps;
              kwargs...)
  which_decomp::String = get(kwargs, :which_decomp, "automatic")
  obs = get(kwargs,:observer, NoObserver())
  quiet::Bool = get(kwargs, :quiet, false)

  # eigsolve kwargs
  eigsolve_tol::Float64   = get(kwargs, :eigsolve_tol, 1e-14)
  eigsolve_krylovdim::Int = get(kwargs, :eigsolve_krylovdim, 3)
  eigsolve_maxiter::Int   = get(kwargs, :eigsolve_maxiter, 1)
  eigsolve_verbosity::Int = get(kwargs, :eigsolve_verbosity, 0)

  # TODO: add support for non-Hermitian DMRG
  # get(kwargs, :ishermitian, true)
  ishermitian::Bool = true

  # TODO: add support for targeting other states with DMRG
  # (such as the state with the largest eigenvalue)
  # get(kwargs, :eigsolve_which_eigenvalue, :SR)
  eigsolve_which_eigenvalue::Symbol = :SR

  # Keyword argument deprecations
  haskey(kwargs,:maxiter) && error("""maxiter keyword has been replace by eigsolve_krylovdim.
                                      Note: compared to the C++ version of ITensor,
                                      setting eigsolve_krylovdim 3 is the same as setting
                                      a maxiter of 2.""")
  haskey(kwargs,:errgoal) && error("errgoal keyword has been replace by eigsolve_tol.")

  psi = copy(psi0)
  N = length(psi)

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
      vals,vecs = eigsolve(PH, phi, 1, eigsolve_which_eigenvalue;
                           ishermitian = ishermitian,
                           tol = eigsolve_tol,
                           krylovdim = eigsolve_krylovdim,
                           maxiter = eigsolve_maxiter)
end
      energy,phi = vals[1],vecs[1]

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

@timeit_debug GLOBAL_TIMER "replacebond!" begin
      spec = replacebond!(psi, b, phi; maxdim = maxdim(sweeps,sw),
                                       mindim = mindim(sweeps,sw),
                                       cutoff = cutoff(sweeps,sw),
                                       noise=noise_mag,
                                       noise_tensor=nt,
                                       dir = dir,
                                       which_decomp = which_decomp)
end

      measure!(obs; energy = energy,
                    psi = psi,
                    bond = b,
                    sweep = sw,
                    half_sweep = ha,
                    spec = spec,
                    quiet = quiet)
    end
    end
    if !quiet
      @printf("After sweep %d energy=%.12f maxlinkdim=%d time=%.3f\n",
              sw, energy, maxlinkdim(psi), sw_time)
    end
    checkdone!(obs; quiet = quiet) && break
  end
  return (energy,psi)
end
