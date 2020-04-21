using SnoopCompile

inf_timing = @snoopi tmin=0.01 include("snoop.jl")
pc = SnoopCompile.parcel(inf_timing)
SnoopCompile.write("tmp", pc)
cp("tmp/precompile_ITensors.jl",
   "precompile.jl";
   force=true)
