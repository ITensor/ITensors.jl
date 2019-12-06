export Sweeps,
       nsweep,
       maxdim,
       mindim,
       cutoff,
       maxdim!,
       mindim!,
       cutoff!,
       sweepnext

mutable struct Sweeps
  nsweep::Int
  maxdim::Vector{Int}
  cutoff::Vector{Float64}
  mindim::Vector{Int}

  function Sweeps(nsw::Int)
    return new(nsw,fill(1,nsw),zeros(nsw),fill(1,nsw))
  end

end

nsweep(sw::Sweeps)::Int = sw.nsweep
Base.length(sw::Sweeps)::Int = sw.nsweep

maxdim(sw::Sweeps,n::Int)::Int = sw.maxdim[n]
mindim(sw::Sweeps,n::Int)::Int = sw.mindim[n]
cutoff(sw::Sweeps,n::Int)::Float64 = sw.cutoff[n]

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

struct SweepNext
  N::Int
end

sweepnext(N::Int)::SweepNext = SweepNext(N)

function Base.iterate(sn::SweepNext,state=(0,1))
  b,ha = state
  if ha==1
    inc = 1
    bstop = sn.N
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

function Base.show(io::IO,
                   sw::Sweeps)
  println(io,"Sweeps")
  for n=1:nsweep(sw)
    @printf(io,"%d cutoff=%.1E, maxdim=%d, mindim=%d\n",n,cutoff(sw,n),maxdim(sw,n),mindim(sw,n))
  end
end
