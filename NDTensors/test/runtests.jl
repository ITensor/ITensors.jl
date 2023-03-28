using Test
using NDTensors

ops = Vector{Function}(undef, 1)
ops[1] = NDTensors.cpu

using Pkg: Pkg

use_cuda = false
if use_cuda
  Pkg.add("CUDA")
  using CUDA
  CUDA.allowscalar()
  if CUDA.functional()
    push!(ops, NDTensors.cu)
  end
end

use_mtl = false
if use_mtl
  Pkg.add("Metal")
  using Metal
  push!(ops, NDTensors.mtl)
  Metal.allowscalar()
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
