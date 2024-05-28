using Adapt: adapt
using KrylovKit: eigsolve
using NDTensors: scalartype, timer
using Printf: @printf
using TupleTools: TupleTools

function permute(
  M::AbstractMPS, ::Tuple{typeof(linkind),typeof(siteinds),typeof(linkind)}
)::typeof(M)
  M̃ = typeof(M)(length(M))
  for n in 1:length(M)
    lₙ₋₁ = linkind(M, n - 1)
    lₙ = linkind(M, n)
    s⃗ₙ = TupleTools.sort(Tuple(siteinds(M, n)); by=plev)
    M̃[n] = ITensors.permute(M[n], filter(!isnothing, (lₙ₋₁, s⃗ₙ..., lₙ)))
  end
  set_ortho_lims!(M̃, ortho_lims(M))
  return M̃
end

function dmrg(H::MPO, psi0::MPS, sweeps::Sweeps; kwargs...)
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  H = permute(H, (linkind, siteinds, linkind))
  PH = ProjMPO(H)
  return dmrg(PH, psi0, sweeps; kwargs...)
end

function dmrg(Hs::Vector{MPO}, psi0::MPS, sweeps::Sweeps; kwargs...)
  for H in Hs
    check_hascommoninds(siteinds, H, psi0)
    check_hascommoninds(siteinds, H, psi0')
  end
  Hs .= permute.(Hs, Ref((linkind, siteinds, linkind)))
  PHS = ProjMPOSum(Hs)
  return dmrg(PHS, psi0, sweeps; kwargs...)
end

function dmrg(H::MPO, Ms::Vector{MPS}, psi0::MPS, sweeps::Sweeps; weight=true, kwargs...)
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  for M in Ms
    check_hascommoninds(siteinds, M, psi0)
  end
  H = permute(H, (linkind, siteinds, linkind))
  Ms .= permute.(Ms, Ref((linkind, siteinds, linkind)))
  if weight <= 0
    error(
      "weight parameter should be > 0.0 in call to excited-state dmrg (value passed was weight=$weight)",
    )
  end
  PMM = ProjMPO_MPS(H, Ms; weight)
  return dmrg(PMM, psi0, sweeps; kwargs...)
end

using NDTensors.TypeParameterAccessors: unwrap_array_type
"""
    dmrg(H::MPO, psi0::MPS; kwargs...)
    dmrg(H::MPO, psi0::MPS, sweeps::Sweeps; kwargs...)

Use the density matrix renormalization group (DMRG) algorithm
to optimize a matrix product state (MPS) such that it is the
eigenvector of lowest eigenvalue of a Hermitian matrix `H`,
represented as a matrix product operator (MPO).

    dmrg(Hs::Vector{MPO}, psi0::MPS; kwargs...)
    dmrg(Hs::Vector{MPO}, psi0::MPS, sweeps::Sweeps; kwargs...)

Use the density matrix renormalization group (DMRG) algorithm
to optimize a matrix product state (MPS) such that it is the
eigenvector of lowest eigenvalue of a Hermitian matrix `H`.
This version of `dmrg` accepts a representation of H as a
Vector of MPOs, `Hs = [H1, H2, H3, ...]` such that `H` is defined
`as H = H1 + H2 + H3 + ...`
Note that this sum of MPOs is not actually computed; rather
the set of MPOs `[H1,H2,H3,..]` is efficiently looped over at
each step of the DMRG algorithm when optimizing the MPS.

    dmrg(H::MPO, Ms::Vector{MPS}, psi0::MPS; weight=1.0, kwargs...)
    dmrg(H::MPO, Ms::Vector{MPS}, psi0::MPS, sweeps::Sweeps; weight=1.0, kwargs...)

Use the density matrix renormalization group (DMRG) algorithm
to optimize a matrix product state (MPS) such that it is the
eigenvector of lowest eigenvalue of a Hermitian matrix `H`,
subject to the constraint that the MPS is orthogonal to each
of the MPS provided in the Vector `Ms`. The orthogonality
constraint is approximately enforced by adding to `H` terms of
the form `w|M1><M1| + w|M2><M2| + ...` where `Ms=[M1, M2, ...]` and
`w` is the "weight" parameter, which can be adjusted through the
optional `weight` keyword argument.

!!! note
    `dmrg` will report the energy of the operator
    `H + w|M1><M1| + w|M2><M2| + ...`, not the operator `H`.
    If you want the expectation value of the MPS eigenstate
    with respect to just `H`, you can compute it yourself with
    an observer or after DMRG is run with `inner(psi', H, psi)`.

The MPS `psi0` is used to initialize the MPS to be optimized.

The number of sweeps of thd DMRG algorithm is controlled by
passing the `nsweeps` keyword argument. The keyword arguments
`maxdim`, `cutoff`, `noise`, and `mindim` can also be passed
to control the cost versus accuracy of the algorithm - see below
for details.

Alternatively the number of sweeps and accuracy parameters can
be passed through a `Sweeps` object, though this interface is
no longer preferred.

Returns:

  - `energy::Number` - eigenvalue of the optimized MPS
  - `psi::MPS` - optimized MPS

Keyword arguments:

  - `nsweeps::Int` - number of "sweeps" of DMRG to perform

Optional keyword arguments:

  - `maxdim` - integer or array of integers specifying the maximum size
     allowed for the bond dimension or rank of the MPS being optimized.
  - `cutoff` - float or array of floats specifying the truncation error cutoff
     or threshold to use for truncating the bond dimension or rank of the MPS.
  - `eigsolve_krylovdim::Int = 3` - maximum dimension of Krylov space used to
     locally solve the eigenvalue problem. Try setting to a higher value if
     convergence is slow or the Hamiltonian is close to a critical point. [^krylovkit]
  - `eigsolve_tol::Number = 1e-14` - Krylov eigensolver tolerance. [^krylovkit]
  - `eigsolve_maxiter::Int = 1` - number of times the Krylov subspace can be
     rebuilt. [^krylovkit]
  - `eigsolve_verbosity::Int = 0` - verbosity level of the Krylov solver.
     Warning: enabling this will lead to a lot of outputs to the terminal. [^krylovkit]
  - `ishermitian=true` - boolean specifying if dmrg should assume the MPO (or more
     general linear operator) represents a Hermitian matrix. [^krylovkit]
  - `noise` - float or array of floats specifying strength of the "noise term"
     to use to aid convergence.
  - `mindim` - integer or array of integers specifying the minimum size of the
     bond dimension or rank, if possible.
  - `outputlevel::Int = 1` - larger outputlevel values make DMRG print more
     information and 0 means no output.
  - `observer` - object implementing the [Observer](@ref observer) interface
     which can perform measurements and stop DMRG early.
  - `write_when_maxdim_exceeds::Int` - when the allowed maxdim exceeds this
     value, begin saving tensors to disk to free RAM memory in large calculations
  - `write_path::String = tempdir()` - path to use to save files to disk
     (to save RAM) when maxdim exceeds the `write_when_maxdim_exceeds` option, if set

[^krylovkit]:

    The `dmrg` function in `ITensors.jl` currently uses the `eigsolve`
    function in `KrylovKit.jl` as the internal the eigensolver.
    See the `KrylovKit.jl` documention on the `eigsolve` function for more details:
    [KrylovKit.eigsolve](https://jutho.github.io/KrylovKit.jl/stable/man/eig/#KrylovKit.eigsolve).
"""
function dmrg(
  PH,
  psi0::MPS,
  sweeps::Sweeps;
  which_decomp=nothing,
  svd_alg=nothing,
  observer=NoObserver(),
  outputlevel=1,
  write_when_maxdim_exceeds=nothing,
  write_path=tempdir(),
  # eigsolve kwargs
  eigsolve_tol=1e-14,
  eigsolve_krylovdim=3,
  eigsolve_maxiter=1,
  eigsolve_verbosity=0,
  eigsolve_which_eigenvalue=:SR,
  ishermitian=true,
)
  if length(psi0) == 1
    error(
      "`dmrg` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.eigsolve`, etc.",
    )
  end

  @debug_check begin
    # Debug level checks
    # Enable with ITensors.enable_debug_checks()
    checkflux(psi0)
    checkflux(PH)
  end

  psi = copy(psi0)
  N = length(psi)
  if !isortho(psi) || orthocenter(psi) != 1
    psi = orthogonalize!(PH, psi, 1)
  end
  @assert isortho(psi) && orthocenter(psi) == 1

  if !isnothing(write_when_maxdim_exceeds)
    if (maxlinkdim(psi) > write_when_maxdim_exceeds) ||
      (maxdim(sweeps, 1) > write_when_maxdim_exceeds)
      PH = disk(PH; path=write_path)
    end
  end
  PH = position!(PH, psi, 1)
  energy = 0.0

  for sw in 1:nsweep(sweeps)
    sw_time = @elapsed begin
      maxtruncerr = 0.0

      if !isnothing(write_when_maxdim_exceeds) &&
        maxdim(sweeps, sw) > write_when_maxdim_exceeds
        if outputlevel >= 2
          println(
            "\nWriting environment tensors do disk (write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim(sweeps, sw) = $(maxdim(sweeps, sw))).\nFiles located at path=$write_path\n",
          )
        end
        PH = disk(PH; path=write_path)
      end

      for (b, ha) in sweepnext(N)
        @debug_check begin
          checkflux(psi)
          checkflux(PH)
        end

        @timeit_debug timer "dmrg: position!" begin
          PH = position!(PH, psi, b)
        end

        @debug_check begin
          checkflux(psi)
          checkflux(PH)
        end

        @timeit_debug timer "dmrg: psi[b]*psi[b+1]" begin
          phi = psi[b] * psi[b + 1]
        end

        @timeit_debug timer "dmrg: eigsolve" begin
          vals, vecs = eigsolve(
            PH,
            phi,
            1,
            eigsolve_which_eigenvalue;
            ishermitian,
            tol=eigsolve_tol,
            krylovdim=eigsolve_krylovdim,
            maxiter=eigsolve_maxiter,
          )
        end

        energy = vals[1]
        ## Right now there is a conversion problem in CUDA.jl where `UnifiedMemory` Arrays are being converted
        ## into `DeviceMemory`. This conversion line is here temporarily to fix that problem when it arises
        ## Adapt is only called when using CUDA backend. CPU will work as implemented previously.
        ## TODO this might be the only place we really need iscu if its not fixed.
        phi = if NDTensors.iscu(phi) && NDTensors.iscu(vecs[1])
          adapt(ITensors.set_eltype(unwrap_array_type(phi), eltype(vecs[1])), vecs[1])
        else
          vecs[1]
        end

        ortho = ha == 1 ? "left" : "right"

        drho = nothing
        if noise(sweeps, sw) > 0
          @timeit_debug timer "dmrg: noiseterm" begin
            # Use noise term when determining new MPS basis.
            # This is used to preserve the element type of the MPS.
            elt = real(scalartype(psi))
            drho = elt(noise(sweeps, sw)) * noiseterm(PH, phi, ortho)
          end
        end

        @debug_check begin
          checkflux(phi)
        end

        @timeit_debug timer "dmrg: replacebond!" begin
          spec = replacebond!(
            PH,
            psi,
            b,
            phi;
            maxdim=maxdim(sweeps, sw),
            mindim=mindim(sweeps, sw),
            cutoff=cutoff(sweeps, sw),
            eigen_perturbation=drho,
            ortho,
            normalize=true,
            which_decomp,
            svd_alg,
          )
        end
        maxtruncerr = max(maxtruncerr, spec.truncerr)

        @debug_check begin
          checkflux(psi)
          checkflux(PH)
        end

        if outputlevel >= 2
          @printf("Sweep %d, half %d, bond (%d,%d) energy=%s\n", sw, ha, b, b + 1, energy)
          @printf(
            "  Truncated using cutoff=%.1E maxdim=%d mindim=%d\n",
            cutoff(sweeps, sw),
            maxdim(sweeps, sw),
            mindim(sweeps, sw)
          )
          @printf(
            "  Trunc. err=%.2E, bond dimension %d\n", spec.truncerr, dim(linkind(psi, b))
          )
          flush(stdout)
        end

        sweep_is_done = (b == 1 && ha == 2)
        measure!(
          observer;
          energy,
          psi,
          projected_operator=PH,
          bond=b,
          sweep=sw,
          half_sweep=ha,
          spec,
          outputlevel,
          sweep_is_done,
        )
      end
    end
    if outputlevel >= 1
      @printf(
        "After sweep %d energy=%s  maxlinkdim=%d maxerr=%.2E time=%.3f\n",
        sw,
        energy,
        maxlinkdim(psi),
        maxtruncerr,
        sw_time
      )
      flush(stdout)
    end
    isdone = checkdone!(observer; energy, psi, sweep=sw, outputlevel)
    isdone && break
  end
  return (energy, psi)
end

function _dmrg_sweeps(;
  nsweeps,
  maxdim=default_maxdim(),
  mindim=default_mindim(),
  cutoff=default_cutoff(Float64),
  noise=default_noise(),
)
  sweeps = Sweeps(nsweeps)
  setmaxdim!(sweeps, maxdim...)
  setmindim!(sweeps, mindim...)
  setcutoff!(sweeps, cutoff...)
  setnoise!(sweeps, noise...)
  return sweeps
end

function dmrg(
  x1,
  x2,
  psi0::MPS;
  nsweeps,
  maxdim=default_maxdim(),
  mindim=default_mindim(),
  cutoff=default_cutoff(Float64),
  noise=default_noise(),
  kwargs...,
)
  return dmrg(
    x1, x2, psi0, _dmrg_sweeps(; nsweeps, maxdim, mindim, cutoff, noise); kwargs...
  )
end

function dmrg(
  x1,
  psi0::MPS;
  nsweeps,
  maxdim=default_maxdim(),
  mindim=default_mindim(),
  cutoff=default_cutoff(Float64),
  noise=default_noise(),
  kwargs...,
)
  return dmrg(x1, psi0, _dmrg_sweeps(; nsweeps, maxdim, mindim, cutoff, noise); kwargs...)
end
