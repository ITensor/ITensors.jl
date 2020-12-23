using Test

@testset "ITensors.jl" begin
  @testset "$filename" for filename in [
    "tagset.jl",
    "smallstring.jl",
    "index.jl",
    "indexset.jl",
    "not.jl",
    "itensor.jl",
    "itensor_slice.jl",
    "itensor_scalar_contract.jl",
    "itensor_combine_contract.jl",
    "broadcast.jl",
    "diagitensor.jl",
    "contract.jl",
    "combiner.jl",
    "debug_checks.jl",
    "trg.jl",
    "ctmrg.jl",
    "iterativesolvers.jl",
    "dmrg.jl",
    "sitetype.jl",
    "phys_site_types.jl",
    "decomp.jl",
    "lattices.jl",
    "mps.jl",
    "mpo.jl",
    "sweeps.jl",
    "sweepnext.jl",
    "autompo.jl",
    "svd.jl",
    "qn.jl",
    "qnindex.jl",
    "qnitensor.jl",
    "qncombiner.jl",
    "qndiagitensor.jl",
    "empty.jl",
    "qnmpo.jl",
    "readwrite.jl",
    "readme.jl",
    "examples.jl",
   ]
    println("Running $filename")
    include(filename)
  end
end
