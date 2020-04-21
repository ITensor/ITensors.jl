
abstract type AbstractObserver end

measure!(o::AbstractObserver; kwargs...) = nothing
checkdone!(o::AbstractObserver; kwargs...) = false

"""
NoObserver is a trivial implementation of an
observer type which can be used as a default
argument for DMRG routines taking an AbstractObserver
"""
struct NoObserver <: AbstractObserver
end

const DMRGMeasurement = Vector{Vector{Float64}}

"""
DMRGObserver is an implementation of an
observer object (<:AbstractObserver) which
implements custom measurements and allows
the `dmrg` function to return early if an
energy convergence criterion is met.
"""
struct DMRGObserver <: AbstractObserver
  ops::Vector{String}
  sites::Vector{<:Index}
  measurements::Dict{String,DMRGMeasurement}
  energies::Vector{Float64}
  truncerrs::Vector{Float64}
  etol::Float64
  minsweeps::Int64

  function DMRGObserver(energy_tol=0.0, 
                        minsweeps=2) 
    new([],[],Dict{String,DMRGMeasurement}(),[],[],energy_tol,minsweeps)
  end

  function DMRGObserver(ops::Vector{String}, 
                        sites::Vector{<:Index};
                        energy_tol=0.0,
                        minsweeps=2)
    measurements = Dict(o => DMRGMeasurement() for o in ops)
    return new(ops,sites,measurements,[],[],energy_tol,minsweeps)
  end
end

measurements(o::DMRGObserver) = o.measurements
energies(o::DMRGObserver) = o.energies
sites(obs::DMRGObserver) = obs.sites
ops(obs::DMRGObserver) = obs.ops
truncerrors(obs::DMRGObserver) = obs.truncerrs

function measurelocalops!(obs::DMRGObserver,
                          wf::ITensor,
                          i::Int)
  for o in ops(obs)
    m = dot(wf, noprime(op(sites(obs),o,i)*wf))
    imag(m)>1e-8 && (@warn "encountered finite imaginary part when measuring $o")
    measurements(obs)[o][end][i]=real(m)
  end
end

function measure!(obs::DMRGObserver;
                  kwargs...)
  half_sweep = kwargs[:half_sweep]
  b = kwargs[:bond]
  energy = kwargs[:energy]
  psi = kwargs[:psi]
  truncerr = truncerror(kwargs[:spec])

  if half_sweep==2
    N = length(psi)

    if b==(N-1)
      for o in ops(obs)
        push!(measurements(obs)[o],zeros(N))
      end
      push!(truncerrors(obs),0.0)
    end

    # when sweeping left the orthogonality center is located
    # at site n=b after the bond update.
    # We want to measure at n=b+1 because there the tensor has been
    # already fully updated (by the right and left pass of the sweep).
    wf = psi[b]*psi[b+1]
    measurelocalops!(obs,wf,b+1)

    if b==1
      push!(energies(obs), energy)
      measurelocalops!(obs,wf,b)
    end
    truncerr > truncerrors(obs)[end] && (truncerrors(obs)[end] = truncerr)
  end
end

function checkdone!(o::DMRGObserver; kwargs...)
  outputlevel = get(kwargs,:outputlevel,false)
  if (length(energies(o)) > o.minsweeps &&
      abs(energies(o)[end] - energies(o)[end-1]) < o.etol)
    outputlevel > 0 && println("Energy difference less than $(o.etol), stopping DMRG")
    return true
  end
  return false
end

