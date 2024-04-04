@eval module $(gensym())
using NDTensors.GradedAxes: gradedisequal, gradedrange, fusion_product
using NDTensors.Sectors: ⊗, Fib, Ising, SU, SU2, U1, Z, quantum_dimension
using Test: @inferred, @test, @testset, @test_throws

@testset "Simple object fusion rules" begin
  @testset "Z{2} fusion rules" begin
    z0 = Z{2}(0)
    z1 = Z{2}(1)

    @test z0 ⊗ z0 == z0
    @test z0 ⊗ z1 == z1
    @test z1 ⊗ z1 == z0
    @test (@inferred z0 ⊗ z0) == z0  # no better way, see Julia PR 23426

    # using GradedAxes interface
    @test gradedisequal(fusion_product(z0, z0), gradedrange([z0 => 1]))
    @test gradedisequal(fusion_product(z0, z1), gradedrange([z1 => 1]))
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

    @test gradedisequal(j1 ⊗ j2, gradedrange([j2 => 1]))
    @test gradedisequal(j2 ⊗ j2, gradedrange([j1 => 1, j3 => 1]))
    @test gradedisequal(j2 ⊗ j3, gradedrange([j2 => 1, j4 => 1]))
    @test gradedisequal(j3 ⊗ j3, gradedrange([j1 => 1, j3 => 1, j5 => 1]))
    @test gradedisequal((@inferred j1 ⊗ j2), gradedrange([j2 => 1]))
    @test (@inferred quantum_dimension(j1 ⊗ j2)) == 2
  end

  @testset "SU{2} fusion rules" begin
    j1 = SU{2}(1)
    j2 = SU{2}(2)
    j3 = SU{2}(3)
    j4 = SU{2}(4)
    j5 = SU{2}(5)

    @test gradedisequal(j1 ⊗ j2, gradedrange([j2 => 1]))
    @test gradedisequal(j2 ⊗ j2, gradedrange([j1 => 1, j3 => 1]))
    @test gradedisequal(j2 ⊗ j3, gradedrange([j2 => 1, j4 => 1]))
    @test gradedisequal(j3 ⊗ j3, gradedrange([j1 => 1, j3 => 1, j5 => 1]))
    @test gradedisequal((@inferred j1 ⊗ j2), gradedrange([j2 => 1]))
  end

  @testset "Fibonacci fusion rules" begin
    ı = Fib("1")
    τ = Fib("τ")

    @test gradedisequal(ı ⊗ ı, gradedrange([ı => 1]))
    @test gradedisequal(ı ⊗ τ, gradedrange([τ => 1]))
    @test gradedisequal(τ ⊗ ı, gradedrange([τ => 1]))
    @test gradedisequal((@inferred τ ⊗ τ), gradedrange([ı => 1, τ => 1]))
    @test (@inferred quantum_dimension(gradedrange([ı => 1, ı => 1]))) == 2.0
  end

  @testset "Ising fusion rules" begin
    ı = Ising("1")
    σ = Ising("σ")
    ψ = Ising("ψ")

    @test gradedisequal(ı ⊗ ı, gradedrange([ı => 1]))
    @test gradedisequal(ı ⊗ σ, gradedrange([σ => 1]))
    @test gradedisequal(σ ⊗ ı, gradedrange([σ => 1]))
    @test gradedisequal(ı ⊗ ψ, gradedrange([ψ => 1]))
    @test gradedisequal(ψ ⊗ ı, gradedrange([ψ => 1]))
    @test gradedisequal(σ ⊗ σ, gradedrange([ı => 1, ψ => 1]))
    @test gradedisequal(σ ⊗ ψ, gradedrange([σ => 1]))
    @test gradedisequal(ψ ⊗ σ, gradedrange([σ => 1]))
    @test gradedisequal(ψ ⊗ ψ, gradedrange([ı => 1]))
    @test gradedisequal((@inferred ψ ⊗ ψ), gradedrange([ı => 1]))
    @test (@inferred quantum_dimension(σ ⊗ σ)) == 2.0
  end
end
@testset "Reducible object fusion rules" begin
  @testset "GradedUnitRange fusion rules" begin
    g1 = gradedrange([U1(1) => 1, U1(2) => 2])
    g2 = gradedrange([U1(-1) => 2, U1(0) => 1, U1(1) => 2])
    @test gradedisequal(
      (@inferred fusion_product(g1, g2)),
      gradedrange([U1(0) => 2, U1(1) => 5, U1(2) => 4, U1(3) => 4]),
    )

    g3 = gradedrange([SU2(0) => 1, SU2(1//2) => 2, SU2(1) => 1])
    g4 = gradedrange([SU2(1//2) => 1, SU2(1) => 2])
    @test gradedisequal(
      (@inferred fusion_product(g3, g4)),
      gradedrange([SU2(0) => 4, SU2(1//2) => 6, SU2(1) => 6, SU2(3//2) => 5, SU2(2) => 2]),
    )

    # test different categories cannot be fused
    @test_throws MethodError fusion_product(g1, g4)
  end

  @testset "Mixed GradedUnitRange - Category fusion rules" begin
    g1 = gradedrange([U1(1) => 1, U1(2) => 2])
    g2 = gradedrange([U1(2) => 1, U1(3) => 2])
    @test gradedisequal((@inferred fusion_product(g1, U1(1))), g2)
    @test gradedisequal((@inferred fusion_product(U1(1), g1)), g2)

    g3 = gradedrange([SU2(0) => 1, SU2(1//2) => 2])
    g4 = gradedrange([SU2(0) => 2, SU2(1//2) => 1, SU2(1) => 2])
    @test gradedisequal((@inferred fusion_product(g3, SU2(1//2))), g4)
    @test gradedisequal((@inferred fusion_product(SU2(1//2), g3)), g4)

    # test different categories cannot be fused
    @test_throws MethodError fusion_product(g1, SU2(1))
    @test_throws MethodError fusion_product(U1(1), g3)
  end
end
end
