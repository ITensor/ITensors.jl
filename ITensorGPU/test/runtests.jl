if VERSION < v"1.7"
  itensorgpu_path = joinpath(pwd(), "..")
  old_path = pop!(LOAD_PATH)
  push!(LOAD_PATH, itensorgpu_path)
  push!(LOAD_PATH, old_path)
end

using CUDA
println("Running ITensorGPU tests with a runtime CUDA version: $(CUDA.runtime_version())")

using ITensorGPU, Test

CUDA.allowscalar(false)
using Pkg
Pkg.add("Combinatorics")
@testset "ITensorGPU.jl" begin
  #@testset "$filename" for filename in ("test_cucontract.jl",)
  #  println("Running $filename with autotune")
  #  cmd = `$(Base.julia_cmd()) -e 'using Pkg; Pkg.activate(".."); Pkg.instantiate(); include("test_cucontract.jl")'`
  #  run(pipeline(setenv(cmd, "CUTENSOR_AUTOTUNE" => 1); stdout=stdout, stderr=stderr))
  #end
  @testset "$filename" for filename in (
    "test_dmrg.jl",
    "test_cuitensor.jl",
    "test_cudiag.jl",
    "test_cudense.jl",
    "test_cucontract.jl",
    "test_cumpo.jl",
    "test_cumps.jl",
    "test_cuiterativesolvers.jl",
    "test_cutruncate.jl",
    #"test_pastaq.jl",
  )
    println("Running $filename")
    include(filename)
  end
end
