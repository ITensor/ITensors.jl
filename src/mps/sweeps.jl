
mutable struct Sweeps
  nsweep::Int
  maxm::Vector{Int}
  cutoff::Vector{Float64}
  minm::Vector{Int}

  function Sweeps(nsw::Int)
    return new(nsw,fill(1,nsw),zeros(nsw),fill(1,nsw))
  end

end

nsweep(sw::Sweeps)::Int = sw.nsweep

maxm(sw::Sweeps,n::Int)::Int = sw.maxm[n]
minm(sw::Sweeps,n::Int)::Int = sw.minm[n]
cutoff(sw::Sweeps,n::Int)::Float64 = sw.cutoff[n]

function maxm!(sw::Sweeps,maxms::Int...)::Nothing
  Nm = length(maxms)
  N = min(nsweep(sw),Nm)
  for i=1:N
    sw.maxm[i] = maxms[i]
  end
  for i=Nm+1:nsweep(sw)
    sw.maxm[i] = maxms[Nm]
  end
end

function minm!(sw::Sweeps,minms::Int...)::Nothing
  Nm = length(minms)
  N = min(nsweep(sw),Nm)
  for i=1:N
    sw.minm[i] = minms[i]
  end
  for i=Nm+1:nsweep(sw)
    sw.minm[i] = minms[Nm]
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

function iterate(sn::SweepNext,state=(0,1))
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

#function sweepnext(b::Int,
#                   ha::Int,
#                   N::Int)::Tuple{Int,Int,Bool}
#  if ha==1
#    inc = 1
#    bstop = N
#  else
#    inc = -1
#    bstop = 0
#  end
#  new_b = b+inc
#  new_ha = ha
#  done = false
#  if new_b==bstop
#    new_b -= inc
#    new_ha += 1
#    (ha==2) && (done = true)
#  end
#  return (new_b,new_ha,done)
#end
