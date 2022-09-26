
abstract type AbstractObserver end

measure!(o::AbstractObserver; kwargs...) = nothing
checkdone!(o::AbstractObserver; kwargs...) = false

"""
NoObserver is a trivial implementation of an
observer type which can be used as a default
argument for DMRG routines taking an AbstractObserver
"""
struct NoObserver <: AbstractObserver end

"""
A DMRGMeasurement object is an alias for `Vector{Vector{Float64}}`,
in other words an array of arrays of real numbers.

Given a DMRGMeasurement `M`,the result for the
measurement on sweep `n` and site `i` as `M[n][i]`.
"""
const DMRGMeasurement = Vector{Vector{Float64}}

"""
DMRGObserver is an implementation of an
observer object (<:AbstractObserver) which
implements custom measurements and allows
the `dmrg` function to return early if an
energy convergence criterion is met.
"""
struct DMRGObserver{T} <: AbstractObserver
  ops::Vector{String}
  sites::Vector{<:Index}
  measurements::Dict{String,DMRGMeasurement}
  energies::Vector{T}
  truncerrs::Vector{Float64}
  etol::Float64
  minsweeps::Int64
end

"""
    DMRGObserver(;energy_tol=0.0,
                  minsweeps=2,
                  energy_type=Float64)

Construct a DMRGObserver by providing the energy
tolerance used for early stopping, and minimum number
of sweeps that must be done.

Optional keyword arguments:

  - energy_tol: if the energy from one sweep to the
    next no longer changes by more than this amount,
    stop after the current sweep
  - minsweeps: do at least this many sweeps
  - energy_type: type to use when storing energies at each step
"""
function DMRGObserver(; energy_tol=0.0, minsweeps=2, energy_type=Float64)
  return DMRGObserver(
    String[],
    Index[],
    Dict{String,DMRGMeasurement}(),
    energy_type[],
    Float64[],
    energy_tol,
    minsweeps,
  )
end

"""
    DMRGObserver(ops::Vector{String}, 
                 sites::Vector{<:Index};
                 energy_tol=0.0,
                 minsweeps=2,
                 energy_type=Float64)

Construct a DMRGObserver, provide an array
of `ops` of operator names which are strings
recognized by the `op` function. Each of
these operators will be measured on every site
during every step of DMRG and the results
recorded inside the DMRGOberver for later
analysis. The array `sites` is the basis
of sites used to define the MPS and MPO for
the DMRG calculation.

Optionally, one can provide an energy
tolerance used for early stopping, and minimum number
of sweeps that must be done.

Optional keyword arguments:

  - energy_tol: if the energy from one sweep to the
    next no longer changes by more than this amount,
    stop after the current sweep
  - minsweeps: do at least this many sweeps
  - energy_type: type to use when storing energies at each step
"""
function DMRGObserver(
  ops::Vector{String},
  sites::Vector{<:Index};
  energy_tol=0.0,
  minsweeps=2,
  energy_type=Float64,
)
  measurements = Dict(o => DMRGMeasurement() for o in ops)
  return DMRGObserver{energy_type}(
    ops, sites, measurements, energy_type[], Float64[], energy_tol, minsweeps
  )
end

"""
    measurements(o::DMRGObserver)

After using a DMRGObserver object `o` within
a DMRG calculation, retrieve a dictionary
of measurement results, with the keys being
operator names and values being DMRGMeasurement
objects.
"""
measurements(o::DMRGObserver) = o.measurements

"""
    energies(o::DMRGObserver)

After using a DMRGObserver object `o` within
a DMRG calculation, retrieve an array of the
energy after each sweep.
"""
energies(o::DMRGObserver) = o.energies

sites(obs::DMRGObserver) = obs.sites

ops(obs::DMRGObserver) = obs.ops

truncerrors(obs::DMRGObserver) = obs.truncerrs

function measurelocalops!(obs::DMRGObserver, wf::ITensor, i::Int)
  for o in ops(obs)
    # Moves to GPU if needed
    oⱼ = adapt(datatype(wf), op(sites(obs), o, i))
    m = dot(wf, apply(oⱼ, wf))
    imag(m) > 1e-8 && (@warn "encountered finite imaginary part when measuring $o")
    measurements(obs)[o][end][i] = real(m)
  end
end

function measure!(obs::DMRGObserver; kwargs...)
  half_sweep = kwargs[:half_sweep]
  b = kwargs[:bond]
  energy = kwargs[:energy]
  psi = kwargs[:psi]
  truncerr = truncerror(kwargs[:spec])

  if half_sweep == 2
    N = length(psi)

    if b == (N - 1)
      for o in ops(obs)
        push!(measurements(obs)[o], zeros(N))
      end
      push!(truncerrors(obs), 0.0)
    end

    # when sweeping left the orthogonality center is located
    # at site n=b after the bond update.
    # We want to measure at n=b+1 because there the tensor has been
    # already fully updated (by the right and left pass of the sweep).
    wf = psi[b] * psi[b + 1]
    measurelocalops!(obs, wf, b + 1)

    if b == 1
      push!(energies(obs), energy)
      measurelocalops!(obs, wf, b)
    end
    truncerr > truncerrors(obs)[end] && (truncerrors(obs)[end] = truncerr)
  end
end

function checkdone!(o::DMRGObserver; kwargs...)
  outputlevel = get(kwargs, :outputlevel, false)
  if (
    length(real(energies(o))) > o.minsweeps &&
    abs(real(energies(o))[end] - real(energies(o))[end - 1]) < o.etol
  )
    outputlevel > 0 && println("Energy difference less than $(o.etol), stopping DMRG")
    return true
  end
  return false
end
