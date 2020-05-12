using Test

@testset "ITensors.jl" begin
  @testset "$filename" for filename in (
    "tagset.jl",
    "smallstring.jl",
    "index.jl",
    "indexset.jl",
    "not.jl",
    "itensor.jl",
    "broadcast.jl",
    "diagitensor.jl",
    "contract.jl",
    "combiner.jl",
    "trg.jl",
    "ctmrg.jl",
    "iterativesolvers.jl",
    "dmrg.jl",
    "tag_types.jl",
    "phys_site_types.jl",
    "decomp.jl",
    "lattices.jl",
    "mps.jl",
    "mpo.jl",
    "sweepnext.jl",
    "autompo.jl",
    "svd.jl",
    "qn.jl",
    "qnindex.jl",
    "qnitensor.jl",
    "qndiagitensor.jl",
    "empty.jl",
    "qnmpo.jl",
    "readwrite.jl",
    "readme.jl",
    "examples.jl",
  )
    println("Running $filename")
    include(filename)
  end
end
