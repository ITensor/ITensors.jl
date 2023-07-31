using Test
using SafeTestsets

println("Passing arguments ARGS=$(ARGS) to test.")

if isempty(ARGS) || "base" in ARGS
  println(
    """\nArguments ARGS = $(ARGS) are empty, or contain `"base"`. Running cpu NDTensors tests.""",
  )
end
if "cuda" in ARGS || "all" in ARGS
  println("""\nArguments ARGS = $(ARGS) contain `"cuda"`. Running NDTensorCUDA tests.""")
  using CUDA
end
if "metal" in ARGS || "all" in ARGS
  println("""\nArguments ARGS = $(ARGS) contain`"metal"`. Running NDTensorMetal tests.""")
  using Metal
end

@safetestset "NDTensors" begin
  @testset "$filename" for filename in [
    "SetParameters.jl",
    "linearalgebra.jl",
    "dense.jl",
    "blocksparse.jl",
    "diagblocksparse.jl",
    "diag.jl",
    "emptynumber.jl",
    "emptystorage.jl",
    "combiner.jl",
  ]
    println("Running $filename")
    include(filename)
  end
  if "cuda" in ARGS || "all" in ARGS
    include(joinpath(pkgdir(NDTensors), "ext", "examples", "NDTensorCUDA.jl"))
  end
end

nothing
