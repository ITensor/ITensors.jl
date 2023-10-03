using NDTensors.TagSets
using NDTensors.InlineStrings
using NDTensors.SmallVectors
using NDTensors.SortedSets
using NDTensors.TagSets

using BenchmarkTools
using Cthulhu
using Profile
using PProf

function main(; profile=false)
  TS = SmallTagSet{10,String31}
  ts1 = TS(["a", "b"])
  ts2 = TS(["b", "c", "d"])

  @btime $TS($("x,y"))

  @show union(ts1, ts2)
  @show intersect(ts1, ts2)
  @show setdiff(ts1, ts2)
  @show symdiff(ts1, ts2)

  @btime union($ts1, $ts2)
  @btime intersect($ts1, $ts2)
  @btime setdiff($ts1, $ts2)
  @btime symdiff($ts1, $ts2)

  @show addtags(ts1, ts2)
  @show commontags(ts1, ts2)
  @show removetags(ts1, ts2)
  @show noncommontags(ts1, ts2)
  @show replacetags(ts1, ["b"], ["c", "d"])

  @btime addtags($ts1, $ts2)
  @btime commontags($ts1, $ts2)
  @btime removetags($ts1, $ts2)
  @btime noncommontags($ts1, $ts2)
  @btime replacetags($ts1, $(["b"]), $(["c", "d"]))

  if profile
    Profile.clear()
    @profile foreach(_ -> TagSet("x,y"; data_type=set_type), 1:1_000_000)
    return pprof()
  end
  return nothing
end
