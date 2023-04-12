using Test
using NDTensors

ops = Vector{Function}(undef, 1)
ops[1] = NDTensors.cpu

@show NDTensors.use_cuda
if NDTensors.use_cuda
  println("Testing with CUDA")
  using CUDA
  CUDA.allowscalar()
  if CUDA.functional()
    push!(ops, NDTensors.cu)
    include(joinpath(pkgdir(NDTensors), "ext", "examples", "NDTensorCUDA.jl"))
  end
end

if NDTensors.use_mtl
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
