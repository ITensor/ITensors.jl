using ITensors
using Test
using Suppressor

include("util.jl")

@testset "SVD Algorithms" begin
  @testset "Matrix With Zero Sing Val" begin
    M = [
      1.0 2.0 5.0 4.0
      1.0 1.0 1.0 1.0
      0.0 0.5 0.5 1.0
      0.0 1.0 1.0 2.0
    ]
    U, S, V = NDTensors.svd_recursive(M)
    @test norm(U * LinearAlgebra.Diagonal(S) * V' - M) < 1E-13
  end

  @testset "Real Matrix" begin
    M = rand(10, 20)
    U, S, V = NDTensors.svd_recursive(M)
    @test norm(U * LinearAlgebra.Diagonal(S) * V' - M) < 1E-12

    M = rand(20, 10)
    U, S, V = NDTensors.svd_recursive(M)
    @test norm(U * LinearAlgebra.Diagonal(S) * V' - M) < 1E-12
  end

  @testset "Cplx Matrix" begin
    M = rand(ComplexF64, 10, 15)
    U, S, V = NDTensors.svd_recursive(M)
    @test norm(U * LinearAlgebra.Diagonal(S) * V' - M) < 1E-13

    M = rand(ComplexF64, 15, 10)
    U, S, V = NDTensors.svd_recursive(M)
    @test norm(U * LinearAlgebra.Diagonal(S) * V' - M) < 1E-13
  end

  @testset "Regression Test 1" begin
    # Implementation of the SVD was giving low
    # accuracy for this case
    M = rand(2, 2, 2, 2)

    M[:, :, 1, 1] = [
      7.713134067177845 -0.16367628720441685
      -1.5253996568409225 1.3577749944302373
    ]

    M[:, :, 2, 1] = [
      0.0 -2.1219889218225276
      -8.320068013774126 0.43565608213298096
    ]

    M[:, :, 1, 2] = [
      0.0 -8.662721825820825
      0.0 -0.46817091771736885
    ]

    M[:, :, 2, 2] = [
      0.0 0.0
      0.0 -8.159570989998151
    ]

    t1 = Index(2, "t1")
    t2 = Index(2, "t2")
    u1 = Index(2, "u1")
    u2 = Index(2, "u2")

    T = itensor(M, t1, t2, u1, u2)

    U, S, V = svd(T, (u1, t1))
    @test norm(U * S * V - T) / norm(T) < 1E-10
  end

  # TODO: remove this test, it takes a long time
  ## @testset "Ill-conditioned matrix" begin
  ##   d = 5000
  ##   i = Index(d, "i")
  ##   T = itensor(make_illconditioned_matrix(dim(i)), i', i)

  ##   @suppress begin
  ##     F = svd(T, i'; alg="divide_and_conquer")
  ##   end
  ##   # Depending on the LAPACK implementation,
  ##   # this sometimes works so don't test it
  ##   #@test isnothing(F)

  ##   # XXX: This fails on Windows, removing for now.
  ##   # F = svd(T, i'; alg="qr_iteration")
  ##   # @test !isnothing(F)
  ##   # @test F.U * F.S * F.V ≈ T
  ## end
end

nothing
