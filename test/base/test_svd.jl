using ITensors
using Test
using Suppressor

include(joinpath(@__DIR__, "utils", "util.jl"))

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

  @testset "svd with empty left or right indices" for space in
                                                      (2, [QN(0, 2) => 1, QN(1, 2) => 1]),
    cutoff in (nothing, 1e-15),
    _eltype in (Float32, Float64, ComplexF32, ComplexF64)

    i = Index(space)
    j = Index(space)
    A = randomITensor(_eltype, i, j)

    U, S, V = svd(A, i, j; cutoff)
    @test eltype(U) <: _eltype
    @test eltype(S) <: real(_eltype)
    @test eltype(V) <: _eltype
    @test U * S * V ≈ A
    @test hassameinds(uniqueinds(U, S), A)
    @test isempty(uniqueinds(V, S))
    @test dim(U) == dim(A)
    @test dim(S) == 1
    @test dim(V) == 1
    @test order(U) == order(A) + 1
    @test order(S) == 2
    @test order(V) == 1

    U, S, V = svd(A, (); cutoff)
    @test eltype(U) <: _eltype
    @test eltype(S) <: real(_eltype)
    @test eltype(V) <: _eltype
    @test U * S * V ≈ A
    @test hassameinds(uniqueinds(V, S), A)
    @test isempty(uniqueinds(U, S))
    @test dim(U) == 1
    @test dim(S) == 1
    @test dim(V) == dim(A)
    @test order(U) == 1
    @test order(S) == 2
    @test order(V) == order(A) + 1

    @test_throws ErrorException svd(A)
  end

  @testset "factorize with empty left or right indices" for space in (
      2, [QN(0, 2) => 1, QN(1, 2) => 1]
    ),
    cutoff in (nothing, 1e-15)

    i = Index(space)
    j = Index(space)
    A = randomITensor(i, j)

    X, Y = factorize(A, i, j; cutoff)
    @test X * Y ≈ A
    @test hassameinds(uniqueinds(X, Y), A)
    @test isempty(uniqueinds(Y, X))
    @test dim(X) == dim(A)
    @test dim(Y) == 1
    @test order(X) == order(A) + 1
    @test order(Y) == 1

    X, Y = factorize(A, (); cutoff)
    @test X * Y ≈ A
    @test hassameinds(uniqueinds(Y, X), A)
    @test isempty(uniqueinds(X, Y))
    @test dim(X) == 1
    @test dim(Y) == dim(A)
    @test order(X) == 1
    @test order(Y) == order(A) + 1

    @test_throws ErrorException factorize(A)
  end

  @testset "svd with empty left and right indices" for cutoff in (nothing, 1e-15)
    A = ITensor(3.4)

    U, S, V = svd(A, (); cutoff)
    @test U * S * V ≈ A
    @test isempty(uniqueinds(U, S))
    @test isempty(uniqueinds(V, S))
    @test dim(U) == 1
    @test dim(S) == 1
    @test dim(V) == 1
    @test order(U) == 1
    @test order(S) == 2
    @test order(V) == 1

    @test_throws ErrorException svd(A)
  end

  @testset "factorize with empty left and right indices" for cutoff in (nothing, 1e-15)
    A = ITensor(3.4)

    X, Y = factorize(A, (); cutoff)
    @test X * Y ≈ A
    @test isempty(uniqueinds(X, Y))
    @test isempty(uniqueinds(Y, X))
    @test dim(X) == 1
    @test dim(Y) == 1
    @test order(X) == 1
    @test order(Y) == 1

    @test_throws ErrorException factorize(A)
  end

  @testset "svd with single precision element type" for eltype in (Float32, ComplexF32),
    space in (2, [QN(0) => 1, QN(1) => 1])

    i = Index(space)
    A = randomITensor(eltype, i', dag(i))
    @test Base.eltype(A) === eltype
    U, S, V = svd(A, i'; maxdim=1)
    @test Base.eltype(U) === eltype
    @test Base.eltype(S) === real(eltype)
    @test Base.eltype(V) === eltype
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
