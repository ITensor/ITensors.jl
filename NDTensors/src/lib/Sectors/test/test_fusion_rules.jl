@eval module $(gensym())
using NDTensors.GradedAxes: fuse_labels, gradedrange
using NDTensors.Sectors: ⊕, ⊗, Fib, Ising, SU, SU2, U1, Z, dimension
using Test: @inferred, @test, @testset

@testset "sum rules" begin

  # test abelian
  q1 = U1(1)
  q2 = U1(2)
  @test q1 ⊕ q2 == gradedrange([q1 => 1, q2 => 1])
  @test q2 ⊕ q1 == gradedrange([q2 => 1, q1 => 1])  # unsorted
  @test q1 ⊕ q1 == gradedrange([q1 => 1, q1 => 1])
  @test dimension(gradedrange([q1 => 1, q2 => 2])) == 3

  # test non-abelian
  j2 = SU2(1//2)
  j3 = SU2(1)
  @test j2 ⊕ j3 == gradedrange([j2 => 1, j3 => 1])
  @test j3 ⊕ j2 == gradedrange([j3 => 1, j2 => 1])  # unsorted
  @test j2 ⊕ j2 == gradedrange([j2 => 1, j2 => 1])
  @test dimension(gradedrange([j2 => 2, j3 => 3])) == 13
end

@testset "fusion rules" begin
  @testset "Z{2} fusion rules" begin
    z0 = Z{2}(0)
    z1 = Z{2}(1)

    @test z0 ⊗ z0 == z0
    @test z0 ⊗ z1 == z1
    @test z1 ⊗ z1 == z0
    @test (@inferred z0 ⊗ z0) == z0  # no better way, see Julia PR 23426

    # using GradedAxes interface
    @test fuse_labels(z0, z0) == z0
    @test fuse_labels(z0, z1) == z1
  end
  @testset "U(1) fusion rules" begin
    q1 = U1(1)
    q2 = U1(2)
    q3 = U1(3)

    @test q1 ⊗ q1 == U1(2)
    @test q1 ⊗ q2 == U1(3)
    @test q2 ⊗ q1 == U1(3)
    @test (@inferred q1 ⊗ q2) == q3  # no better way, see Julia PR 23426
  end
  @testset "SU2 fusion rules" begin
    j1 = SU2(0)
    j2 = SU2(1//2)
    j3 = SU2(1)
    j4 = SU2(3//2)
    j5 = SU2(2)

    @test j1 ⊗ j2 == gradedrange([j2 => 1])
    @test j2 ⊗ j2 == j1 ⊕ j3
    @test j2 ⊗ j3 == j2 ⊕ j4
    @test j3 ⊗ j3 == j1 ⊕ j3 ⊕ j5
    @test (@inferred j1 ⊗ j2) == gradedrange([j2 => 1])
  end

  @testset "SU{2} fusion rules" begin
    j1 = SU{2}(1)
    j2 = SU{2}(2)
    j3 = SU{2}(3)
    j4 = SU{2}(4)
    j5 = SU{2}(5)

    @test j1 ⊗ j2 == gradedrange([j2 => 1])
    @test j2 ⊗ j2 == j1 ⊕ j3
    @test j2 ⊗ j3 == j2 ⊕ j4
    @test j3 ⊗ j3 == j1 ⊕ j3 ⊕ j5
    @test (@inferred j1 ⊗ j2) == gradedrange([j2 => 1])
  end

  @testset "Fibonacci fusion rules" begin
    ı = Fib("1")
    τ = Fib("τ")

    @test ı ⊗ ı == gradedrange([ı => 1])
    @test ı ⊗ τ == gradedrange([τ => 1])
    @test τ ⊗ ı == gradedrange([τ => 1])
    @test τ ⊗ τ == ı ⊕ τ
    @test (@inferred τ ⊗ τ) == ı ⊕ τ
  end

  @testset "Ising fusion rules" begin
    ı = Ising("1")
    σ = Ising("σ")
    ψ = Ising("ψ")

    @test ı ⊗ ı == gradedrange([ı => 1])
    @test ı ⊗ σ == gradedrange([σ => 1])
    @test σ ⊗ ı == gradedrange([σ => 1])
    @test ı ⊗ ψ == gradedrange([ψ => 1])
    @test ψ ⊗ ı == gradedrange([ψ => 1])
    @test σ ⊗ σ == ı ⊕ ψ
    @test σ ⊗ ψ == gradedrange([σ => 1])
    @test ψ ⊗ σ == gradedrange([σ => 1])
    @test ψ ⊗ ψ == gradedrange([ı => 1])
    @test (@inferred ψ ⊗ ψ) == gradedrange([ı => 1])
  end
end
end
