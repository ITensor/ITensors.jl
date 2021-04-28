using SnoopCompile

# Note: in order to generate a new precompile.jl file,
# you need to comment out the lines:
# ```
# include("../precompile/precompile.jl")
# _precompile_()
# ```
# in src/ITensors.jl (otherwise many functions will
# be precompiled and SnoopCompile won't think they
# need to get precompiled).
#
# Also, snooping on dmrg itself causes problems in
# precompilation. Explicitly add the command:
# ```
# precompile(Tuple{typeof(dmrg),MPO,MPS,Sweeps})
# ```
# to precompile.jl after it is generated.

inf_timing = @snoopi tmin = 0.01 include("snoop/snoop.jl")
pc = SnoopCompile.parcel(inf_timing)
SnoopCompile.write("tmp", pc)
cp("tmp/precompile_ITensors.jl", "precompile.jl"; force=true)
