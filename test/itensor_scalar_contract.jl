using Test
using ITensors
using Random

Random.seed!(1234)

@testset "Test contractions with scalar-like ITensors" begin
  i = Index(2, "i")
  j = Index(2, "j")
  k = Index(2, "k")
  α = Index(1, "α")

  is = (i, j, k)

  A = randomITensor(is..., dag(α))
  B = ITensor(2, α, α', α'')

  C = A * B
  @test C ≈ B[1, 1, 1] * A * ITensor(1, inds(B))

  C = emptyITensor(is..., α', α'')
  C .= A .* B
  @test C ≈ B[1, 1, 1] * A * ITensor(1, inds(B))

  C = emptyITensor(shuffle([(is..., α', α'')...])...)
  C .= A .* B
  @test C ≈ B[1, 1, 1] * A * ITensor(1, inds(B))
end

@testset "NaN in-place contraction bug regression test" begin
  BlasFloats = (Float32, Float64, ComplexF32, ComplexF64)
  @testset "Scalar contract, no permutation" for ElA in BlasFloats, ElB in BlasFloats
    i = Index(2, "i")
    j = Index(3, "j")
    α = Index(1, "α")
    A = randomITensor(ElA, i, j, α')
    B = randomITensor(ElB, dag(α)', α)
    R = ITensor(promote_type(ElA, ElB), i, j, α)

    R .= NaN
    @test any(isnan, R)

    R .= A .* B
    @test !any(isnan, R)
    @test array(R) ≈ array(A) * array(B)[]

    R .= NaN
    @test any(isnan, R)

    R .= B .* A
    @test !any(isnan, R)
    @test array(R) ≈ array(A) * array(B)[]
  end

  @testset "Scalar contraction, permutation" for ElA in BlasFloats, ElB in BlasFloats
    i = Index(2, "i")
    j = Index(3, "j")
    α = Index(1, "α")
    A = randomITensor(ElA, i, j, α')
    B = randomITensor(ElB, dag(α)', α)
    R = ITensor(promote_type(ElA, ElB), j, i, α)

    R .= NaN
    @test any(isnan, R)

    R .= A .* B
    @test !any(isnan, R)
    @test array(R) ≈ permutedims(array(A), (2, 1, 3)) * array(B)[]

    R .= NaN
    @test any(isnan, R)

    R .= B .* A
    @test !any(isnan, R)
    @test array(R) ≈ permutedims(array(A), (2, 1, 3)) * array(B)[]
  end

  @testset "General contraction, permutation" for ElA in BlasFloats, ElB in BlasFloats
    i = Index(2, "i")
    j = Index(3, "j")
    α = Index(2, "α")
    A = randomITensor(ElA, i, j, α')
    B = randomITensor(ElB, dag(α)', α)
    R = ITensor(promote_type(ElA, ElB), j, i, α)

    R .= NaN
    @test any(isnan, R)

    R .= A .* B
    @test !any(isnan, R)
    @test reshape(array(R), 6, 2) ≈
      reshape(permutedims(array(A), (2, 1, 3)), 6, 2) * array(B)

    R .= NaN
    @test any(isnan, R)

    R .= B .* A
    @test !any(isnan, R)
    @test reshape(array(R), 6, 2) ≈
      reshape(permutedims(array(A), (2, 1, 3)), 6, 2) * array(B)
  end
end
