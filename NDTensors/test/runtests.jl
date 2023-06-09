using NDTensors
using Test
using SafeTestsets

test_args = copy(ARGS)
println("Passing arguments ARGS=$(test_args) to test.")

if isempty(test_args) || "base" in test_args
  println(
    """\nArguments ARGS = $(test_args) are empty, or contain `"base"`. Running cpu NDTensors tests.""",
  )
end
if "cuda" in test_args || "all" in test_args
  using CUDA
  println(
    """\nArguments ARGS = $(test_args) contain `"cuda"`. Running NDTensorCUDA tests."""
  )
end
if "metal" in test_args || "all" in test_args
  using Metal
  println(
    """\nArguments ARGS = $(test_args) contain`"metal"`. Running NDTensorMetal tests."""
  )
end

@safetestset "NDTensors" begin
  @testset "$filename" for filename in [
    "SetParameters.jl",
    "linearalgebra.jl",
    "dense.jl",
    "blocksparse.jl",
    "diag.jl",
    "emptynumber.jl",
    "emptystorage.jl",
    "combiner.jl",
  ]
    println("Running $filename")
    include(filename)
  end
end


nothing
