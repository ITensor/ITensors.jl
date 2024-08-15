@eval module $(gensym())
using NDTensors.GradedAxes:
  dual, fusion_product, gradedisequal, gradedrange, flip, tensor_product
using NDTensors.Sectors:
  ⊗, Fib, Ising, O2, SU, SU2, U1, Z, block_dimensions, quantum_dimension, trivial
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

    # test different input number
    @test gradedisequal(fusion_product(z0), gradedrange([z0 => 1]))
    @test gradedisequal(fusion_product(z0, z0, z0), gradedrange([z0 => 1]))
    @test gradedisequal(fusion_product(z0, z0, z0, z0), gradedrange([z0 => 1]))
    @test (@inferred block_dimensions(gradedrange([z1 => 1]))) == [1]
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

  @testset "O2 fusion rules" begin
    s0e = O2(0)
    s0o = O2(-1)
    s12 = O2(1//2)
    s1 = O2(1)

    @test gradedisequal((@inferred s0e ⊗ s0e), gradedrange([s0e => 1]))
    @test gradedisequal((@inferred s0o ⊗ s0e), gradedrange([s0o => 1]))
    @test gradedisequal((@inferred s0o ⊗ s0e), gradedrange([s0o => 1]))
    @test gradedisequal((@inferred s0o ⊗ s0o), gradedrange([s0e => 1]))

    @test gradedisequal((@inferred s0e ⊗ s12), gradedrange([s12 => 1]))
    @test gradedisequal((@inferred s0o ⊗ s12), gradedrange([s12 => 1]))
    @test gradedisequal((@inferred s12 ⊗ s0e), gradedrange([s12 => 1]))
    @test gradedisequal((@inferred s12 ⊗ s0o), gradedrange([s12 => 1]))
    @test gradedisequal((@inferred s12 ⊗ s1), gradedrange([s12 => 1, O2(3//2) => 1]))
    @test gradedisequal((@inferred s12 ⊗ s12), gradedrange([s0o => 1, s0e => 1, s1 => 1]))

    @test (@inferred quantum_dimension(s0o ⊗ s1)) == 2
    @test (@inferred block_dimensions(s0o ⊗ s1)) == [2]
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
    @test (@inferred block_dimensions(j1 ⊗ j2)) == [2]

    @test gradedisequal(fusion_product(j2), gradedrange([j2 => 1]))
    @test gradedisequal(fusion_product(j2, j1), gradedrange([j2 => 1]))
    @test gradedisequal(fusion_product(j2, j1, j1), gradedrange([j2 => 1]))
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
  @testset "Trivial GradedUnitRange" begin
    g1 = gradedrange([U1(0) => 1])
    g2 = gradedrange([SU2(0) => 1])
    @test gradedisequal(trivial(g1), g1)
    @test gradedisequal(trivial(dual(g1)), g1)  # trivial returns nondual
    @test gradedisequal(trivial(typeof(g2)), g2)
  end
  @testset "GradedUnitRange abelian tensor/fusion product" begin
    g1 = gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 2])
    g2 = gradedrange([U1(-2) => 2, U1(0) => 1, U1(1) => 2])

    @test gradedisequal(flip(dual(g1)), gradedrange([U1(1) => 1, U1(0) => 1, U1(-1) => 2]))
    @test (@inferred block_dimensions(g1)) == [1, 1, 2]

    gt = gradedrange([
      U1(-3) => 2,
      U1(-2) => 2,
      U1(-1) => 4,
      U1(-1) => 1,
      U1(0) => 1,
      U1(1) => 2,
      U1(0) => 2,
      U1(1) => 2,
      U1(2) => 4,
    ])
    gf = gradedrange([
      U1(-3) => 2, U1(-2) => 2, U1(-1) => 5, U1(0) => 3, U1(1) => 4, U1(2) => 4
    ])
    @test gradedisequal((@inferred tensor_product(g1, g2)), gt)
    @test gradedisequal((@inferred fusion_product(g1, g2)), gf)

    gtd1 = gradedrange([
      U1(-1) => 2,
      U1(-2) => 2,
      U1(-3) => 4,
      U1(1) => 1,
      U1(0) => 1,
      U1(-1) => 2,
      U1(2) => 2,
      U1(1) => 2,
      U1(0) => 4,
    ])
    gfd1 = gradedrange([
      U1(-3) => 4, U1(-2) => 2, U1(-1) => 4, U1(0) => 5, U1(1) => 3, U1(2) => 2
    ])
    @test gradedisequal((@inferred tensor_product(dual(g1), g2)), gtd1)
    @test gradedisequal((@inferred fusion_product(dual(g1), g2)), gfd1)

    gtd2 = gradedrange([
      U1(1) => 2,
      U1(2) => 2,
      U1(3) => 4,
      U1(-1) => 1,
      U1(0) => 1,
      U1(1) => 2,
      U1(-2) => 2,
      U1(-1) => 2,
      U1(0) => 4,
    ])
    gfd2 = gradedrange([
      U1(-2) => 2, U1(-1) => 3, U1(0) => 5, U1(1) => 4, U1(2) => 2, U1(3) => 4
    ])
    @test gradedisequal((@inferred tensor_product(g1, dual(g2))), gtd2)
    @test gradedisequal((@inferred fusion_product(g1, dual(g2))), gfd2)

    gtd = gradedrange([
      U1(3) => 2,
      U1(2) => 2,
      U1(1) => 4,
      U1(1) => 1,
      U1(0) => 1,
      U1(-1) => 2,
      U1(0) => 2,
      U1(-1) => 2,
      U1(-2) => 4,
    ])
    gfd = gradedrange([
      U1(-2) => 4, U1(-1) => 4, U1(0) => 3, U1(1) => 5, U1(2) => 2, U1(3) => 2
    ])
    @test gradedisequal((@inferred tensor_product(dual(g1), dual(g2))), gtd)
    @test gradedisequal((@inferred fusion_product(dual(g1), dual(g2))), gfd)

    # test different (non-product) categories cannot be fused
    @test_throws MethodError fusion_product(gradedrange([Z{2}(0) => 1]), g1)
    @test_throws MethodError tensor_product(gradedrange([Z{2}(0) => 1]), g2)
  end

  @testset "GradedUnitRange non-abelian fusion rules" begin
    g3 = gradedrange([SU2(0) => 1, SU2(1//2) => 2, SU2(1) => 1])
    g4 = gradedrange([SU2(1//2) => 1, SU2(1) => 2])
    g34 = gradedrange([
      SU2(1//2) => 1,
      SU2(0) => 2,
      SU2(1) => 2,
      SU2(1//2) => 1,
      SU2(3//2) => 1,
      SU2(1) => 2,
      SU2(1//2) => 4,
      SU2(3//2) => 4,
      SU2(0) => 2,
      SU2(1) => 2,
      SU2(2) => 2,
    ])

    @test gradedisequal(tensor_product(g3, g4), g34)

    @test gradedisequal(dual(flip(g3)), g3)  # trivial for SU(2)
    @test gradedisequal(
      (@inferred fusion_product(g3, g4)),
      gradedrange([SU2(0) => 4, SU2(1//2) => 6, SU2(1) => 6, SU2(3//2) => 5, SU2(2) => 2]),
    )
    @test (@inferred block_dimensions(g3)) == [1, 4, 3]

    # test dual on non self-conjugate non-abelian representations
    s1 = SU{3}((0, 0))
    f3 = SU{3}((1, 0))
    c3 = SU{3}((1, 1))
    ad8 = SU{3}((2, 1))

    g5 = gradedrange([s1 => 1, f3 => 1])
    g6 = gradedrange([s1 => 1, c3 => 1])
    @test gradedisequal(dual(flip(g5)), g6)
    @test gradedisequal(
      fusion_product(g5, g6), gradedrange([s1 => 2, f3 => 1, c3 => 1, ad8 => 1])
    )
    @test gradedisequal(
      fusion_product(dual(g5), g6),
      gradedrange([s1 => 1, f3 => 1, c3 => 2, SU{3}((2, 2)) => 1]),
    )
    @test gradedisequal(
      fusion_product(g5, dual(g6)),
      gradedrange([s1 => 1, f3 => 2, c3 => 1, SU{3}((2, 0)) => 1]),
    )
    @test gradedisequal(
      fusion_product(dual(g5), dual(g6)), gradedrange([s1 => 2, f3 => 1, c3 => 1, ad8 => 1])
    )
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
