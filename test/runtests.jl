using Test

if Threads.nthreads() == 1
  @testset "ITensors.jl" begin
    @testset "$filename" for filename in [
      "ContractionSequenceOptimization/runtests.jl",
      "ITensorChainRules/runtests.jl",
      "LazyApply/runtests.jl",
      "Ops/runtests.jl",
      "tagset.jl",
      "smallstring.jl",
      "symmetrystyle.jl",
      "index.jl",
      "indexset.jl",
      "ndtensors.jl",
      "not.jl",
      "inference.jl",
      "itensor_scalar.jl",
      "itensor.jl",
      "indices.jl",
      "itensor_slice.jl",
      "itensor_scalar_contract.jl",
      "itensor_combine_contract.jl",
      "broadcast.jl",
      "emptyitensor.jl",
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
      "fermions.jl",
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
elseif Threads.nthreads() > 1
  @testset "ITensors.jl threaded" begin
    @testset "$filename" for filename in ["threading.jl"]
      println("Running $filename")
      include(filename)
    end
  end
end
