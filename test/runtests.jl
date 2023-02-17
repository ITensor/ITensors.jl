using ITensors
using Test

ITensors.Strided.disable_threads()
ITensors.BLAS.set_num_threads(1)
ITensors.disable_threaded_blocksparse()

@testset "ITensors tests" begin
  if isempty(ARGS) || "all" in ARGS || "base" in ARGS
    dirs = [
      "LazyApply",
      "Ops",
      "base",
      "threading",
      "ContractionSequenceOptimization",
      "ITensorChainRules",
      "ITensorNetworkMaps",
    ]
    for dir in dirs
      println("\nTest $(@__DIR__)/$(dir)")
      @time include(joinpath(@__DIR__, dir, "runtests.jl"))
    end
  end
  if isempty(ARGS) || "all" in ARGS || "mps" in ARGS
    dir = "ITensorLegacyMPS"
    println("\nTest $(@__DIR__)/$(dir)")
    @time include(joinpath(@__DIR__, dir, "runtests.jl"))
  end
end
