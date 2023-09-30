using NDTensors.TagSets
using NDTensors.InlineStrings
using NDTensors.SmallVectors
using NDTensors.SortedSets

using BenchmarkTools
using Profile
using PProf

function main()
  T = String31
  V = SmallVector{20,T}
  # V = Vector{T}
  S = SortedSet{T,V}
  # const TS = TagSet{T,S}
  TS = S

  ts1 = TS(["a", "b"])
  ts2 = TS(["c", "d"])

  @btime union($ts1, $ts2)

  ## using ProfileView
  ## @profview addtags(ts1, ts2)
  ## @profview addtags(ts1, ts2)
  ## @profview foreach(_ -> addtags(ts1, ts2), 1:100_000)

  # Collect a profile
  Profile.clear()
  @profile foreach(_ -> union(ts1, ts2), 1:10_000_000)

  # Export pprof profile and open interactive profiling web interface.
  pprof()
end
