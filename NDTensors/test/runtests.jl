using Test
using NDTensors

ops = Vector{Function}(undef, 0)
test_args = copy(ARGS)

println("Passing arguments ARGS=$(test_args) to test.")
if isempty(test_args) || "base" in test_args
  println(
  """\nArguments ARGS = $(test_args) are empty, or contain `"base"`. Running cpu NDTensors tests.""",
)
push!(ops, NDTensors.cpu)
end


if "cuda" in test_args || "all" in test_args
  println(
  """\nArguments ARGS = $(test_args) contain `"cuda"`. Running NDTensorCUDA tests.""",
)
  using Pkg
  Pkg.add("CUDA")
  using CUDA
  CUDA.allowscalar()
  if CUDA.functional()
    push!(ops, NDTensors.cu)
    include(joinpath(pkgdir(NDTensors), "ext", "examples", "NDTensorCUDA.jl"))
  end
end

if "metal" in test_args || "all" in test_args
  println(
  """\nArguments ARGS = $(test_args) contain`"metal"`. Running NDTensorMetal tests.""",
)
  using Metal
  push!(ops, NDTensors.mtl)
  Metal.allowscalar()
  include(joinpath(pkgdir(NDTensors), "ext", "examples", "NDTensorMetal.jl"))
end

@testset "NDTensors" begin
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
