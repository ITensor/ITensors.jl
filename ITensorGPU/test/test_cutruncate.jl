using ITensors,
  ITensorGPU,
  LinearAlgebra, # For tr()
  CUDA,
  Test

# gpu tests!
@testset "cutrunctate" begin
  @test ITensorGPU.truncate!(CUDA.zeros(Float64, 10)) == (0.0, 0.0, CUDA.zeros(Float64, 1))
  trunc = ITensorGPU.truncate!(
    CuArray([1.0, 0.5, 0.4, 0.1, 0.05]); absoluteCutoff=true, cutoff=0.2
  )
  @test trunc[1] ≈ 0.15
  @test trunc[2] ≈ 0.25
  @test Array(trunc[3]) == [1.0, 0.5, 0.4]
  trunc = ITensorGPU.truncate!(
    CuArray([0.4, 0.26, 0.19, 0.1, 0.05]); relativeCutoff=true, cutoff=0.2
  )
  @test trunc[1] ≈ 0.15
  @test trunc[2] ≈ 0.145
  @test Array(trunc[3]) == [0.4, 0.26, 0.19]
  trunc = ITensorGPU.truncate!(
    CuArray([0.4, 0.26, 0.19, 0.1, 0.05] / 2); relativeCutoff=true, cutoff=0.2
  )
  @test trunc[1] ≈ 0.15
  @test trunc[2] ≈ 0.145 / 2
  @test Array(trunc[3]) == [0.4, 0.26, 0.19] / 2
end # End truncate test
