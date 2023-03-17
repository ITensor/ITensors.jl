using ITensors,
  ITensorGPU,
  LinearAlgebra, # For tr()
  CUDA,
  Test

# gpu tests!
@testset "cutrunctate" begin
  @testset for T in (Float32, Float64)
    @test ITensorGPU.truncate!(CUDA.zeros(T, 10)) == (zero(T), zero(T), CUDA.zeros(T, 1))
    trunc = ITensorGPU.truncate!(
      CuArray(T[1.0, 0.5, 0.4, 0.1, 0.05]); absoluteCutoff=true, cutoff=T(0.2)
    )
    @test trunc[1] ≈ T(0.15)
    @test trunc[2] ≈ T(0.25)
    @test Array(trunc[3]) == T[1.0, 0.5, 0.4]

    trunc = ITensorGPU.truncate!(
      CuArray(T[1.0, 0.5, 0.4, 0.1, 0.05]); absoluteCutoff=true, maxdim=3
    )
    @test trunc[1] ≈ T(0.15)
    @test trunc[2] ≈ T(0.25)
    @test Array(trunc[3]) == T[1.0, 0.5, 0.4]

    trunc = ITensorGPU.truncate!(
      CuArray(T[1.0, 0.5, 0.4, 0.1, 0.05]); absoluteCutoff=true, maxdim=3, cutoff=T(0.07)
    )
    @test trunc[1] ≈ T(0.15)
    @test trunc[2] ≈ T(0.25)
    @test Array(trunc[3]) == T[1.0, 0.5, 0.4]

    trunc = ITensorGPU.truncate!(
      CuArray(T[0.4, 0.26, 0.19, 0.1, 0.05]); relativeCutoff=true, cutoff=T(0.2)
    )
    @test trunc[1] ≈ T(0.15)
    @test trunc[2] ≈ T(0.145)
    @test Array(trunc[3]) == T[0.4, 0.26, 0.19]

    trunc = ITensorGPU.truncate!(
      CuArray(T[0.4, 0.26, 0.19, 0.1, 0.05]); relativeCutoff=true, maxdim=3
    )
    @test trunc[1] ≈ T(0.15)
    @test trunc[2] ≈ T(0.145)
    @test Array(trunc[3]) == T[0.4, 0.26, 0.19]

    trunc = ITensorGPU.truncate!(
      CuArray(T[0.4, 0.26, 0.19, 0.1, 0.05]); relativeCutoff=true, maxdim=3, cutoff=T(0.07)
    )
    @test trunc[1] ≈ T(0.15)
    @test trunc[2] ≈ T(0.145)
    @test Array(trunc[3]) == T[0.4, 0.26, 0.19]

    trunc = ITensorGPU.truncate!(
      CuArray(convert(Vector{T}, [0.4, 0.26, 0.19, 0.1, 0.05] / 2));
      relativeCutoff=true,
      cutoff=T(0.2),
    )
    @test trunc[1] ≈ T(0.15)
    @test trunc[2] ≈ T(0.145 / 2)
    @test Array(trunc[3]) == convert(Vector{T}, [0.4, 0.26, 0.19] / 2)
  end
end # End truncate test
