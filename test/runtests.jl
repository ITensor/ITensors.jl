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
    @time include(joinpath(@__DIR__, dir, "runtests.jl"))
  end

  # TODO: Make optional (if `Threads.nthreads() > 1`)
  @time include(joinpath(@__DIR__, "threading", "runtests.jl"))

  # TODO: Make optional
  @time include(joinpath(@__DIR__, "ITensorLegacyMPS", "runtests.jl"))
end
