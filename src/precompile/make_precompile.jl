using SnoopCompile

inf_timing = @snoopi tmin=0.01 include("snoop.jl")
pc = SnoopCompile.parcel(inf_timing)
SnoopCompile.write(".", pc)
