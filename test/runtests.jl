using Test

@testset "Tensors and ITensors" begin
  @testset "Tensors.jl" begin
    filename = "Tensors/runtests.jl"
    println("Running $filename")
    include(filename)
  end
  @testset "ITensors.jl" begin
    @testset "$filename" for filename in (
      "tagset.jl",
      "smallstring.jl",
      "index.jl",
      "indexset.jl",
      "not.jl",
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
      "qnindex.jl",
      "itensor_blocksparse.jl",
      "itensor_diagblocksparse.jl",
      "readwrite.jl",
      "readme.jl",
      "examples.jl",
    )
      println("Running $filename")
      include(filename)
    end
  end
end
