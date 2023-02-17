using ITensors
using Test

ITensors.Strided.disable_threads()
ITensors.BLAS.set_num_threads(1)
ITensors.disable_threaded_blocksparse()

@testset "ITensors tests" begin
  # Make a copy, since it seems that one of the ITensors base
  # tests modifies `ARGS` and makes them empty.
  test_args = copy(ARGS)
  println("Passed arguments ARGS = $(test_args) to tests.")
  if isempty(test_args) || "all" in test_args || "base" in test_args
    println(
      """\nArguments ARGS = $(test_args) are empty, or contain `"all"` or `"base"`. Running base (non-MPS/MPO) ITensors tests.""",
    )
    dirs = [
      "LazyApply",
      "Ops",
      "base",
      "threading",
      "ContractionSequenceOptimization",
      "ITensorChainRules",
      "ITensorNetworkMaps",
    ]
    @time for dir in dirs
      println("\nTest $(@__DIR__)/$(dir)")
      @time include(joinpath(@__DIR__, dir, "runtests.jl"))
    end
  end
  if isempty(test_args) || "all" in test_args || "mps" in test_args
    println(
      """\nArguments ARGS = $(test_args) are empty, or contain `"all"` or `"mps"`. Running MPS/MPO ITensors tests.""",
    )
    dir = "ITensorLegacyMPS"
    println("\nTest $(@__DIR__)/$(dir)")
    @time include(joinpath(@__DIR__, dir, "runtests.jl"))
  end
end
