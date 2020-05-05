
import .NDTensors: mindim

"""
A Sweeps objects holds information
about the various parameters controlling
a density matrix renormalization group (DMRG)
or similar matrix product state (MPS) calculation.

For a Sweeps object `sw` the available
parameters are:
* `nsweep(sw)` -- the number of sweeps to do
* `maxdim(sw,n)` -- maximum MPS bond dimension for sweep n
* `mindim(sw,n)` -- minimum MPS bond dimension for sweep n
* `cutoff(sw,n)` -- truncation error cutoff for sweep n
* `noise(sw,n)` -- noise term coefficient for sweep n
"""
mutable struct Sweeps
  nsweep::Int
  maxdim::Vector{Int}
  cutoff::Vector{Float64}
  mindim::Vector{Int}
  noise::Vector{Float64}

  function Sweeps(nsw::Int)
    return new(nsw,fill(1,nsw),zeros(nsw),fill(1,nsw), zeros(nsw))
  end

end

"""
    nsweep(sw::Sweeps)
    length(sw::Sweeps)

Obtain the number of sweeps parameterized
by this sweeps object.
"""
nsweep(sw::Sweeps)::Int = sw.nsweep

Base.length(sw::Sweeps)::Int = sw.nsweep

"""
    maxdim(sw::Sweeps,n::Int)

Maximum MPS bond dimension allowed by the
Sweeps object `sw` during sweep `n`
"""
maxdim(sw::Sweeps,n::Int)::Int = sw.maxdim[n]

"""
    mindim(sw::Sweeps,n::Int)

Minimum MPS bond dimension allowed by the
Sweeps object `sw` during sweep `n`
"""
mindim(sw::Sweeps,n::Int)::Int = sw.mindim[n]

"""
    cutoff(sw::Sweeps,n::Int)

Truncation error cutoff setting of the
Sweeps object `sw` during sweep `n`
"""
cutoff(sw::Sweeps,n::Int)::Float64 = sw.cutoff[n]

"""
    noise(sw::Sweeps,n::Int)

Noise term coefficient setting of the
Sweeps object `sw` during sweep `n`
"""
noise(sw::Sweeps,n::Int)::Float64  = sw.noise[n]

"""
    maxdim!(sw::Sweeps,maxdims::Int...)

Set the maximum MPS bond dimension for each
sweep by providing up to `nsweep(sw)` values.
If fewer values are provided, the last value
is repeated for the remaining sweeps.
"""
function maxdim!(sw::Sweeps,maxdims::Int...)::Nothing
  Nm = length(maxdims)
  N = min(nsweep(sw),Nm)
  for i=1:N
    sw.maxdim[i] = maxdims[i]
  end
  for i=Nm+1:nsweep(sw)
    sw.maxdim[i] = maxdims[Nm]
  end
end

"""
    mindim!(sw::Sweeps,maxdims::Int...)

Set the minimum MPS bond dimension for each
sweep by providing up to `nsweep(sw)` values.
If fewer values are provided, the last value
is repeated for the remaining sweeps.
"""
function mindim!(sw::Sweeps,mindims::Int...)::Nothing
  Nm = length(mindims)
  N = min(nsweep(sw),Nm)
  for i=1:N
    sw.mindim[i] = mindims[i]
  end
  for i=Nm+1:nsweep(sw)
    sw.mindim[i] = mindims[Nm]
  end
end

"""
    cutoff!(sw::Sweeps,maxdims::Int...)

Set the MPS truncation error used for each
sweep by providing up to `nsweep(sw)` values.
If fewer values are provided, the last value
is repeated for the remaining sweeps.
"""
function cutoff!(sw::Sweeps,cutoffs::Float64...)::Nothing
  Nm = length(cutoffs)
  N = min(nsweep(sw),Nm)
  for i=1:N
    sw.cutoff[i] = cutoffs[i]
  end
  for i=Nm+1:nsweep(sw)
    sw.cutoff[i] = cutoffs[Nm]
  end
end

"""
    noise!(sw::Sweeps,maxdims::Int...)

Set the noise-term coefficient used for each
sweep by providing up to `nsweep(sw)` values.
If fewer values are provided, the last value
is repeated for the remaining sweeps.
"""
function noise!(sw::Sweeps,noises::Float64...)::Nothing
  Nm = length(noises)
  N = min(nsweep(sw),Nm)
  for i=1:N
    sw.noise[i] = noises[i]
  end
  for i=Nm+1:nsweep(sw)
    sw.noise[i] = noises[Nm]
  end
end

function Base.show(io::IO,
                   sw::Sweeps)
  println(io,"Sweeps")
  for n=1:nsweep(sw)
    @printf(io,"%d cutoff=%.1E, maxdim=%d, mindim=%d, noise=%.1E\n",n,cutoff(sw,n),maxdim(sw,n),mindim(sw,n),noise(sw,n))
  end
end

struct SweepNext
  N::Int
  ncenter::Int
end

"""
    sweepnext(N::Int; ncenter::Int=2)

Returns an iterable object that evaluates
to tuples of the form `(b,ha)` where `b`
is the bond number and `ha` is the half-sweep
number. Takes an optional named argument 
`ncenter` for use with an n-site MPS or DMRG
algorithm, with a default of 2-site.
"""
function sweepnext(N::Int;ncenter::Int=2)::SweepNext 
  if ncenter < 0
    error("ncenter must be non-negative")
  end
  return SweepNext(N,ncenter)
end
   
function Base.iterate(sn::SweepNext,state=(0,1))
  b,ha = state
  if ha==1
    inc = 1
    bstop = sn.N-sn.ncenter+2
  else
    inc = -1
    bstop = 0
  end
  new_b = b+inc
  new_ha = ha
  done = false
  if new_b==bstop
    new_b -= inc
    new_ha += 1
    if ha==2
      return nothing
    end
  end
  return ((new_b,new_ha),(new_b,new_ha))
end

