
IndexSet_ignore_missing(is::Union{Index, Nothing}...) =
  IndexSet(filter(i -> i isa Index, is))

function permute(M::AbstractMPS, ::Tuple{typeof(linkind), typeof(siteinds), typeof(linkind)})
  M̃ = MPO(length(M))
  for n in 1:length(M)
    lₙ₋₁ = linkind(M, n-1)
    lₙ = linkind(M, n)
    s⃗ₙ = IndexSet(sort(Tuple(siteinds(M, n)); by = plev))
    M̃[n] = permute(M[n], IndexSet_ignore_missing(lₙ₋₁, s⃗ₙ..., lₙ))
  end
  set_ortho_lims!(M̃, ortho_lims(M))
  return M̃
end

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
function dmrg(H::MPO, psi0::MPS, sweeps::Sweeps; kwargs...)
  # Permute the indices to have a better memory layout
  # and minimize permutations
  H = permute(H, (linkind, siteinds, linkind))
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
function dmrg(Hs::Vector{MPO}, psi0::MPS, sweeps::Sweeps; kwargs...)
  Hs .= permute.(Hs, Ref((linkind, siteinds, linkind)))
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
function dmrg(H::MPO, Ms::Vector{MPS}, psi0::MPS, sweeps::Sweeps; kwargs...)
  H = permute(H, (linkind, siteinds, linkind))
  Ms .= permute.(Ms, Ref((linkind, siteinds, linkind)))
  weight = get(kwargs,:weight,1.0)
  PMM = ProjMPO_MPS(H,Ms;weight=weight)
  return dmrg(PMM,psi0,sweeps;kwargs...)
end


function dmrg(PH, psi0::MPS, sweeps::Sweeps; kwargs...)
  @debug_check begin
    # Debug level checks
    # Enable with ITensors.enable_debug_checks()
    checkflux(psi0)
    checkflux(PH)
  end

  which_decomp::Union{String, Nothing} = get(kwargs, :which_decomp, nothing)
  svd_alg::String = get(kwargs, :svd_alg, "divide_and_conquer")
  obs = get(kwargs, :observer, NoObserver())
  outputlevel::Int = get(kwargs, :outputlevel, 1)

  # eigsolve kwargs
  eigsolve_tol::Float64 = get(kwargs, :eigsolve_tol, 1e-14)
  eigsolve_krylovdim::Int = get(kwargs, :eigsolve_krylovdim, 3)
  eigsolve_maxiter::Int = get(kwargs, :eigsolve_maxiter, 1)
  eigsolve_verbosity::Int = get(kwargs, :eigsolve_verbosity, 0)

  # TODO: add support for non-Hermitian DMRG
  ishermitian::Bool = get(kwargs, :ishermitian, true)

  # TODO: add support for targeting other states with DMRG
  # (such as the state with the largest eigenvalue)
  # get(kwargs, :eigsolve_which_eigenvalue, :SR)
  eigsolve_which_eigenvalue::Symbol = :SR

  # TODO: use this as preferred syntax for passing arguments
  # to eigsolve
  #default_eigsolve_args = (tol = 1e-14, krylovdim = 3, maxiter = 1,
  #                         verbosity = 0, ishermitian = true,
  #                         which_eigenvalue = :SR)
  #eigsolve = get(kwargs, :eigsolve, default_eigsolve_args)

  # Keyword argument deprecations
  if haskey(kwargs, :maxiter)
    error("""maxiter keyword has been replaced by eigsolve_krylovdim.
             Note: compared to the C++ version of ITensor,
             setting eigsolve_krylovdim 3 is the same as setting
             a maxiter of 2.""")
  end

  if haskey(kwargs, :errgoal)
    error("errgoal keyword has been replaced by eigsolve_tol.")
  end

  if haskey(kwargs, :quiet)
    error("quiet keyword has been replaced by outputlevel")
  end

  psi = copy(psi0)
  N = length(psi)

  position!(PH, psi0, 1)
  energy = 0.0

  for sw=1:nsweep(sweeps)
    sw_time = @elapsed begin

    for (b, ha) in sweepnext(N)

      @debug_check begin
        checkflux(psi)
        checkflux(PH)
      end

      @timeit_debug timer "dmrg: position!" begin
      position!(PH, psi, b)
      end

      @debug_check begin
        checkflux(psi)
        checkflux(PH)
      end

      @timeit_debug timer "dmrg: psi[b]*psi[b+1]" begin
      phi = psi[b] * psi[b+1]
      end

      @timeit_debug timer "dmrg: eigsolve" begin
      vals, vecs = eigsolve(PH, phi, 1, eigsolve_which_eigenvalue;
                            ishermitian = ishermitian,
                            tol = eigsolve_tol,
                            krylovdim = eigsolve_krylovdim,
                            maxiter = eigsolve_maxiter)
      end
      energy, phi = vals[1], vecs[1]

      ortho = ha == 1 ? "left" : "right"

      drho = nothing
      if noise(sweeps, sw) > 0.0
        # Use noise term when determining new MPS basis
        drho = noise(sweeps, sw) * noiseterm(PH,phi,ortho)
      end

      @debug_check begin
        checkflux(phi)
      end

      @timeit_debug timer "dmrg: replacebond!" begin
      spec = replacebond!(psi, b, phi; maxdim = maxdim(sweeps, sw),
                                       mindim = mindim(sweeps, sw),
                                       cutoff = cutoff(sweeps, sw),
                                       eigen_perturbation = drho,
                                       ortho = ortho,
                                       normalize = true,
                                       which_decomp = which_decomp,
                                       svd_alg = svd_alg)
      end

      @debug_check begin
        checkflux(psi)
        checkflux(PH)
      end


      if outputlevel >= 2
        @printf("Sweep %d, half %d, bond (%d,%d) energy=%.12f\n",sw,ha,b,b+1,energy)
        @printf("(Truncated using cutoff=%.1E maxdim=%d mindim=%d)\n",
                cutoff(sweeps, sw),maxdim(sweeps, sw),mindim(sweeps, sw))
        @printf("Trunc. err=%.1E, bond dimension %d\n\n",spec.truncerr,dim(linkind(psi,b)))
      end

      measure!(obs; energy = energy,
                    psi = psi,
                    bond = b,
                    sweep = sw,
                    half_sweep = ha,
                    spec = spec,
                    outputlevel = outputlevel)
    end
    end
    if outputlevel >= 1
      @printf("After sweep %d energy=%.12f maxlinkdim=%d time=%.3f\n",
              sw, energy, maxlinkdim(psi), sw_time)
    end
    isdone = checkdone!(obs;energy=energy,
                            psi=psi,
                            sweep=sw,
                            outputlevel=outputlevel) 

    isdone && break
  end
  return (energy, psi)
end

