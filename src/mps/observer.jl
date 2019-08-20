export DMRGObserver, measurements, energies

"""
      DMRGStepInfo
  Container for passing information about
  DMRG step to an observer object.

  # Fields:
  - dir::Bool - `false` if sweeping is currently towards the right (forward part of sweep)
                 and `true` otherwise
  - sweepnum::Int - current sweep number
  - energy::Float64 - current variational energy
  - bond::Int - last bond on which DMRG update was performed
  """
struct DMRGStepInfo
  dir::Bool
  bond::Int64
  sweepnum::Int64
  energy::Float64
  # TODO: also need to pass here truncerr and singular values
  # once those are returned from replaceBond!
end


sweepdir(si::DMRGStepInfo) = (si.dir ? "left" : "right")
sweepnum(si::DMRGStepInfo) = si.sweepnum
getenergy(si::DMRGStepInfo) = si.energy
bond(si::DMRGStepInfo) = si.bond

abstract type AbstractObserver end

measure!(o::AbstractObserver, args...) = nothing
checkdone(o::AbstractObserver, args...; kwargs...) = false

struct NoObserver <: AbstractObserver
end

struct DMRGObserver <: AbstractObserver
  ops_::Vector{String}
  sites_::SiteSet
  measurements::Dict{String, Vector{Vector{Float64}} }
  energies::Vector{Float64}
  etol::Float64
  minsweeps::Int64

  DMRGObserver(etol::Real=0, minsweeps::Int=2) = new([],SiteSet(),
                                                     Dict{String,Vector{Vector{Float64}}}(),
                                                     [],etol,minsweeps)

  function DMRGObserver(ops::Vector{String}, sites::SiteSet,etol::Real=0,
                        minsweeps::Int=2)
    measurements = Dict(o => Vector{Float64}[] for o in ops)
    return new(ops,sites,measurements,[],etol,minsweeps)
  end
end

measurements(o::DMRGObserver) = o.measurements

energies(o::DMRGObserver) = o.energies

op(obs::DMRGObserver,O,i) = op(obs.sites_,O,i)

function _measure_local_ops!(obs::DMRGObserver,psi::MPS,i)
  for o in obs.ops_
    m = dot(prime(psi[ i ],"Site"), op(obs, o, i)*psi[i])
    imag(m)>1e-8 && (@warn "encountered finite imaginary part when measuring $o")
    obs.measurements[o][end][i]=real(m)
  end
end

function measure!(obs::DMRGObserver, psi::MPS, si::DMRGStepInfo)
  if sweepdir(si)=="left"
    if bond(si)==length(psi)-1
      for o in obs.ops_
        push!(obs.measurements[o],zeros(length(psi)))
      end
    end
    # when sweeping left the orthogonality center is located at site n= bond(si) after the bond update.
    # We want to measure at n=bond(si)+1 because there the tensor has been
    # already fully updated (by the right and left pass of the sweep).
    orthogonalize!(psi,bond(si)+1)
    _measure_local_ops!(obs,psi,bond(si)+1)
    orthogonalize!(psi,bond(si))

    if bond(si)==1
      push!(obs.energies, getenergy(si))
      _measure_local_ops!(obs,psi,bond(si))
    end
  end
end

function checkdone(o::DMRGObserver ; quiet=false)
  if (length(energies(o))>o.minsweeps &&
      abs(energies(o)[end] - energies(o)[end-1]) < o.etol)
    !quiet && println("Energy difference less than $(o.etol), stopping DMRG")
    return true
  else
    return false
  end
end

