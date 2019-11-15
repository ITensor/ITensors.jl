export AbstractObserver,
       measure!,
       checkdone!,
       NoObserver,
       DMRGObserver,
       measurements,
       energies,
       truncerrors


abstract type AbstractObserver end

measure!(o::AbstractObserver; kwargs...) = nothing
checkdone!(o::AbstractObserver; kwargs...) = false

struct NoObserver <: AbstractObserver
end

const DMRGMeasurement = Vector{Vector{Float64}}

struct DMRGObserver <: AbstractObserver
  ops::Vector{String}
  sites::Vector{Index}
  measurements::Dict{String,DMRGMeasurement}
  energies::Vector{Float64}
  truncerrs::Vector{Float64}
  etol::Float64
  minsweeps::Int64

  function DMRGObserver(etol::Real=0, 
                        minsweeps::Int=2) 
    new([],[],Dict{String,DMRGMeasurement}(),[],[],etol,minsweeps)
  end

  function DMRGObserver(ops::Vector{String}, 
                        sites::Vector{Index},
                        etol::Real=0,
                        minsweeps::Int=2)
    measurements = Dict(o => DMRGMeasurement() for o in ops)
    return new(ops,sites,measurements,[],[],etol,minsweeps)
  end
end

measurements(o::DMRGObserver) = o.measurements
energies(o::DMRGObserver) = o.energies
sites(obs::DMRGObserver) = obs.sites
ops(obs::DMRGObserver) = obs.ops
truncerrors(obs::DMRGObserver) = obs.truncerrs

function measureLocalOps!(obs::DMRGObserver,
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

    if b==N-1
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
    measureLocalOps!(obs,wf,b+1)

    if b==1
      push!(energies(obs), energy)
      measureLocalOps!(obs,wf,b)
    end
    truncerr > truncerrors(obs)[end] && (truncerrors(obs)[end] = truncerr)
  end
end

function checkdone!(o::DMRGObserver; kwargs...)
  quiet = get(kwargs,:quiet,false)
  if (length(energies(o)) > o.minsweeps &&
      abs(energies(o)[end] - energies(o)[end-1]) < o.etol)
    !quiet && println("Energy difference less than $(o.etol), stopping DMRG")
    return true
  end
  return false
end

