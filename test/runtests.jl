using ITensors, Test

@testset "ITensors.jl" begin
    @testset "$filename" for filename in (
        "test_tagset.jl",
        "test_smallstring.jl",
        "test_index.jl",
        "test_indexset.jl",
        "test_itensor_dense.jl",
        "test_itensor_diag.jl",
        "test_contract.jl",
        "test_combiner.jl",
        "test_readwrite.jl",
        "test_trg.jl",
        "test_ctmrg.jl",
        "test_iterativesolvers.jl",
        "test_dmrg.jl",
        "test_tag_types.jl",
        "test_phys_site_types.jl",
        "test_decomp.jl",
        "test_lattices.jl",
        "test_mps.jl",
        "test_mpo.jl",
        "test_autompo.jl",
        "test_svd.jl",
        "test_qn.jl",
    )
      println("Running $filename")
      include(filename)
    end
end
