using ITensors
using Test

ITensors.Strided.disable_threads()
ITensors.BLAS.set_num_threads(1)
ITensors.disable_threaded_blocksparse()

@testset "ITensors tests" begin
  println("Passed arguments ARGS = $(ARGS) to tests.")
  if isempty(ARGS) || "all" in ARGS || "base" in ARGS
    println("""\nArguments ARGS = $(ARGS) are empty, or contain `"all"` or `"base"`. Running base (non-MPS/MPO) ITensors tests.""")
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
    println(
      """\nArguments ARGS = $(ARGS) are empty, or contain `"all"` or `"mps"`. Running MPS/MPO ITensors tests.""",
    )
    dir = "ITensorLegacyMPS"
    println("\nTest $(@__DIR__)/$(dir)")
    @time include(joinpath(@__DIR__, dir, "runtests.jl"))
  end
end
