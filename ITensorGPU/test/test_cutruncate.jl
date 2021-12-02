using ITensors,
  ITensorGPU,
  LinearAlgebra, # For tr()
  CUDA,
  Test

# gpu tests!
@testset "cutrunctate" begin
  @test ITensorGPU.truncate!(CUDA.zeros(Float64, 10)) == (0.0, 0.0, CUDA.zeros(Float64, 1))
  trunc = ITensorGPU.truncate!(
    CuArray([1.0, 0.5, 0.1, 0.05]); absoluteCutoff=true, cutoff=0.2
  )
  @test trunc[1] ≈ 0.15
  @test trunc[2] ≈ 0.3
  @test trunc[3] == CuArray([1.0, 0.5])
  trunc = ITensorGPU.truncate!(CuArray([0.5, 0.4, 0.1]); relativeCutoff=true, cutoff=0.2)
  @test trunc[1] ≈ 0.1
  @test trunc[2] ≈ 0.45
  @test trunc[3] == CuArray([0.5])
end # End truncate test
