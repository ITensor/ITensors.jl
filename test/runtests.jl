using Test

@testset "ITensors tests" begin
  if isempty(ARGS) || "all" in ARGS || "basics" in ARGS
    dirs = [
      "LazyApply",
      "Ops",
      "basics",
      "ContractionSequenceOptimization",
      "ITensorChainRules",
      "ITensorNetworkMaps",
    ]
    for dir in dirs
      println("\nTest $(@__DIR__)/$(dir)")
      @time include(joinpath(@__DIR__, dir, "runtests.jl"))
    end
  end
  if isempty(ARGS) || "all" in ARGS || "threading" in ARGS
    dir = "threading"
    println("\nTest $(@__DIR__)/$(dir)")
    @time include(joinpath(@__DIR__, dir, "runtests.jl"))
  end
  if isempty(ARGS) || "all" in ARGS || "mps" in ARGS
    dir = "ITensorLegacyMPS"
    println("\nTest $(@__DIR__)/$(dir)")
    @time include(joinpath(@__DIR__, dir, "runtests.jl"))
  end
end
