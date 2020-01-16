using ITensors, Test

@testset "ITensors.jl" begin
    @testset "$filename" for filename in (
        "Tensors/runtests.jl",
        "tagset.jl",
        "smallstring.jl",
        "index.jl",
        "indexset.jl",
        "itensor_dense.jl",
        "itensor_diag.jl",
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
        "autompo.jl",
        "svd.jl",
        "qn.jl",
        "readme.jl",
        "readwrite.jl",
    )
      println("Running $filename")
      include(filename)
    end
end
nothing
