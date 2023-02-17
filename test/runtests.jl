using Test

@testset "ITensors tests" begin
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

  # TODO: Make optional (if `Threads.nthreads() > 1`)
  dir = "threading"
  println("\nTest $(@__DIR__)/$(dir)")
  @time include(joinpath(@__DIR__, dir, "runtests.jl"))

  # TODO: Make optional
  dir = "ITensorLegacyMPS"
  println("\nTest $(@__DIR__)/$(dir)")
  @time include(joinpath(@__DIR__, dir, "runtests.jl"))
end
