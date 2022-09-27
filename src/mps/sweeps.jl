
"""
A Sweeps objects holds information
about the various parameters controlling
a density matrix renormalization group (DMRG)
or similar matrix product state (MPS) calculation.

For a Sweeps object `sw` the available
parameters are:

  - `nsweep(sw)` -- the number of sweeps to do
  - `maxdim(sw,n)` -- maximum MPS bond dimension for sweep n
  - `mindim(sw,n)` -- minimum MPS bond dimension for sweep n
  - `cutoff(sw,n)` -- truncation error cutoff for sweep n
  - `noise(sw,n)` -- noise term coefficient for sweep n
"""
mutable struct Sweeps
  nsweep::Int
  maxdim::Vector{Int}
  cutoff::Vector{Float64}
  mindim::Vector{Int}
  noise::Vector{Float64}

  function Sweeps(nsw::Int; maxdim=typemax(Int), cutoff=1E-16, mindim=1, noise=0.0)
    sw = new(nsw, fill(typemax(Int), nsw), fill(1E-16, nsw), fill(1, nsw), fill(0.0, nsw))
    setmaxdim!(sw, maxdim...)
    setmindim!(sw, mindim...)
    setcutoff!(sw, cutoff...)
    setnoise!(sw, noise...)
    return sw
  end
end

Sweeps() = Sweeps(0)

"""
    Sweeps(d::AbstractMatrix)

    Sweeps(nsweep::Int, d::AbstractMatrix)

Make a sweeps object from a matrix of input values.
The first row should be strings that define which
variables are being set ("maxdim", "cutoff", "mindim",
and "noise").

If the number of sweeps are not specified, they
are determined from the size of the input matrix.

# Examples

```julia
julia > Sweeps(
  [
    "maxdim" "mindim" "cutoff" "noise"
    50 10 1e-12 1E-7
    100 20 1e-12 1E-8
    200 20 1e-12 1E-10
    400 20 1e-12 0
    800 20 1e-12 1E-11
    800 20 1e-12 0
  ],
)
Sweeps
1cutoff = 1.0E-12, maxdim = 50, mindim = 10, noise = 1.0E-07
2cutoff = 1.0E-12, maxdim = 100, mindim = 20, noise = 1.0E-08
3cutoff = 1.0E-12, maxdim = 200, mindim = 20, noise = 1.0E-10
4cutoff = 1.0E-12, maxdim = 400, mindim = 20, noise = 0.0E+00
5cutoff = 1.0E-12, maxdim = 800, mindim = 20, noise = 1.0E-11
6cutoff = 1.0E-12, maxdim = 800, mindim = 20, noise = 0.0E+00
```
"""
function Sweeps(nsw::Int, d::AbstractMatrix)
  sw = Sweeps(nsw)
  vars = d[1, :]
  for (n, var) in enumerate(vars)
    inputs = d[2:end, n]
    if var == "maxdim"
      maxdim!(sw, inputs...)
    elseif var == "cutoff"
      cutoff!(sw, inputs...)
    elseif var == "mindim"
      mindim!(sw, inputs...)
    elseif var == "noise"
      noise!(sw, float.(inputs)...)
    else
      error("Sweeps object does not have the field $var")
    end
  end
  return sw
end

Sweeps(d::AbstractMatrix) = Sweeps(size(d, 1) - 1, d)

"""
    nsweep(sw::Sweeps)
    length(sw::Sweeps)

Obtain the number of sweeps parameterized
by this sweeps object.
"""
nsweep(sw::Sweeps)::Int = sw.nsweep

Base.length(sw::Sweeps)::Int = sw.nsweep

Base.isempty(sw::Sweeps)::Bool = (sw.nsweep == 0)

"""
    maxdim(sw::Sweeps,n::Int)

Maximum MPS bond dimension allowed by the
Sweeps object `sw` during sweep `n`
"""
maxdim(sw::Sweeps, n::Int)::Int = sw.maxdim[n]

"""
    mindim(sw::Sweeps,n::Int)

Minimum MPS bond dimension allowed by the
Sweeps object `sw` during sweep `n`
"""
mindim(sw::Sweeps, n::Int)::Int = sw.mindim[n]

"""
    cutoff(sw::Sweeps,n::Int)

Truncation error cutoff setting of the
Sweeps object `sw` during sweep `n`
"""
cutoff(sw::Sweeps, n::Int)::Float64 = sw.cutoff[n]

"""
    noise(sw::Sweeps,n::Int)

Noise term coefficient setting of the
Sweeps object `sw` during sweep `n`
"""
noise(sw::Sweeps, n::Int)::Float64 = sw.noise[n]

get_maxdims(sw::Sweeps) = sw.maxdim
get_mindims(sw::Sweeps) = sw.mindim
get_cutoffs(sw::Sweeps) = sw.cutoff
get_noises(sw::Sweeps) = sw.noise

"""
    maxdim!(sw::Sweeps,maxdims::Int...)

Set the maximum MPS bond dimension for each
sweep by providing up to `nsweep(sw)` values.
If fewer values are provided, the last value
is repeated for the remaining sweeps.
"""
function setmaxdim!(sw::Sweeps, maxdims::Int...)::Nothing
  mdims = collect(maxdims)
  for i in 1:nsweep(sw)
    sw.maxdim[i] = get(mdims, i, maxdims[end])
  end
end
maxdim!(sw::Sweeps, maxdims::Int...) = setmaxdim!(sw, maxdims...)

"""
    mindim!(sw::Sweeps,maxdims::Int...)

Set the minimum MPS bond dimension for each
sweep by providing up to `nsweep(sw)` values.
If fewer values are provided, the last value
is repeated for the remaining sweeps.
"""
function setmindim!(sw::Sweeps, mindims::Int...)::Nothing
  mdims = collect(mindims)
  for i in 1:nsweep(sw)
    sw.mindim[i] = get(mdims, i, mindims[end])
  end
end
mindim!(sw::Sweeps, mindims::Int...) = setmindim!(sw, mindims...)

"""
    cutoff!(sw::Sweeps,maxdims::Int...)

Set the MPS truncation error used for each
sweep by providing up to `nsweep(sw)` values.
If fewer values are provided, the last value
is repeated for the remaining sweeps.
"""
function setcutoff!(sw::Sweeps, cutoffs::Real...)::Nothing
  cuts = collect(cutoffs)
  for i in 1:nsweep(sw)
    sw.cutoff[i] = get(cuts, i, cutoffs[end])
  end
end
cutoff!(sw::Sweeps, cutoffs::Real...) = setcutoff!(sw, cutoffs...)

"""
    noise!(sw::Sweeps,maxdims::Int...)

Set the noise-term coefficient used for each
sweep by providing up to `nsweep(sw)` values.
If fewer values are provided, the last value
is repeated for the remaining sweeps.
"""
function setnoise!(sw::Sweeps, noises::Real...)::Nothing
  nvals = collect(noises)
  for i in 1:nsweep(sw)
    sw.noise[i] = get(nvals, i, noises[end])
  end
end
noise!(sw::Sweeps, noises::Real...) = setnoise!(sw, noises...)

function Base.show(io::IO, sw::Sweeps)
  println(io, "Sweeps")
  for n in 1:nsweep(sw)
    @printf(
      io,
      "%d cutoff=%.1E, maxdim=%d, mindim=%d, noise=%.1E\n",
      n,
      cutoff(sw, n),
      maxdim(sw, n),
      mindim(sw, n),
      noise(sw, n)
    )
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
function sweepnext(N::Int; ncenter::Int=2)::SweepNext
  if ncenter < 0
    error("ncenter must be non-negative")
  end
  return SweepNext(N, ncenter)
end

function Base.iterate(sn::SweepNext, state=(0, 1))
  b, ha = state
  if ha == 1
    inc = 1
    bstop = sn.N - sn.ncenter + 2
  else
    inc = -1
    bstop = 0
  end
  new_b = b + inc
  new_ha = ha
  done = false
  if new_b == bstop
    new_b -= inc
    new_ha += 1
    if ha == 2
      return nothing
    end
  end
  return ((new_b, new_ha), (new_b, new_ha))
end
