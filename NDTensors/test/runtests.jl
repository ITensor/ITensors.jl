using Test
using NDTensors
use_cuda = false
if use_cuda
  using CUDA
end

ops = Vector{Function}(undef, 1)
ops[1] = NDTensors.cpu
@require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
  if CUDA.functional()
    push!(ops, cu)
  end
end

@testset "NDTensors" begin
  @testset "$filename" for filename in [
    # "SetParameters.jl",
    # "linearalgebra.jl",
    "dense.jl",
  #   "blocksparse.jl",
  #   "diag.jl",
  #   "emptynumber.jl",
  #   "emptystorage.jl",
  #   "combiner.jl",
   ]
    println("Running $filename")
    include(filename)
  end
end

nothing
