using Test
using NDTensors

ops = Vector{Function}(undef, 1)
ops[1] = NDTensors.cpu

using Pkg: Pkg

use_cuda = true
NDTensors.allow_ndtensorcuda(use_cuda)
if use_cuda
  using CUDA
  CUDA.allowscalar()
  if CUDA.functional()
    push!(ops, NDTensors.cu)
  end
end

use_mtl = false
if use_mtl
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
