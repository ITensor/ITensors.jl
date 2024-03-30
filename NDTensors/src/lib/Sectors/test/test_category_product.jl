@eval module $(gensym())
using NDTensors.Sectors:
  ×, ⊗, CategoryProduct, Fib, Ising, SU, SU2, U1, Z, categories, sector, quantum_dimension
using NDTensors.GradedAxes: dual, gradedrange
using Test: @inferred, @test, @testset, @test_broken, @test_throws

@testset "Test Named Category Products" begin
  @testset "Construct from × of NamedTuples" begin
    s = (A=U1(1),) × (B=Z{2}(0),)
    @test length(categories(s)) == 2
    @test categories(s)[:A] == U1(1)
    @test categories(s)[:B] == Z{2}(0)
    @test (@inferred quantum_dimension(s)) == 1
    @test dual(s) == (A=U1(-1),) × (B=Z{2}(0),)

    s = (A=U1(1),) × (B=SU2(2),)
    @test length(categories(s)) == 2
    @test categories(s)[:A] == U1(1)
    @test categories(s)[:B] == SU2(2)
    @test (@inferred quantum_dimension(s)) == 5
    @test dual(s) == (A=U1(-1),) × (B=SU2(2),)

    s = s × (C=Ising("ψ"),)
    @test length(categories(s)) == 3
    @test categories(s)[:C] == Ising("ψ")
    @test (@inferred quantum_dimension(s)) == 5.0
    @test dual(s) == (A=U1(-1),) × (B=SU2(2),) × (C=Ising("ψ"),)
  end

  @testset "Construct from Pairs" begin
    s = sector("A" => U1(2))
    @test length(categories(s)) == 1
    @test categories(s)[:A] == U1(2)
    @test s == sector(; A=U1(2))
    @test (@inferred quantum_dimension(s)) == 1
    @test dual(s) == sector("A" => U1(-2))

    s = sector("B" => Ising("ψ"), :C => Z{2}(1))
    @test length(categories(s)) == 2
    @test categories(s)[:B] == Ising("ψ")
    @test categories(s)[:C] == Z{2}(1)
    @test (@inferred quantum_dimension(s)) == 1.0
  end

  @testset "Empty category" begin
    s = sector()
    @test dual(s) == s
    @test s × s == s
    @test s ⊗ s == s
    @test (@inferred quantum_dimension(s)) == 0
  end

  @testset "Abelian fusion rules" begin
    q00 = sector()
    q10 = sector(; A=U1(1))
    q01 = sector(; B=U1(1))
    q11 = sector(; A=U1(1), B=U1(1))

    @test q10 ⊗ q10 == sector(; A=U1(2))
    @test q01 ⊗ q00 == q01
    @test q00 ⊗ q01 == q01
    @test q10 ⊗ q01 == q11
    @test q11 ⊗ q11 == sector(; A=U1(2), B=U1(2))
  end

  @testset "U(1) × SU(2) conventional" begin
    q0 = sector()
    q0h = sector(; J=SU2(1//2))
    q10 = (N=U1(1),) × (J=SU2(0),)
    # Put names in reverse order sometimes:
    q1h = (J=SU2(1//2),) × (N=U1(1),)
    q11 = (N=U1(1),) × (J=SU2(1),)
    q20 = sector(; N=U1(2))
    q2h = (N=U1(2),) × (J=SU2(1//2),)
    q21 = (N=U1(2),) × (J=SU2(1),)
    q22 = (N=U1(2),) × (J=SU2(2),)

    @test q1h ⊗ q1h == gradedrange([q20 => 1, q21 => 1])
    @test q10 ⊗ q1h == gradedrange([q2h => 1])
    @test q0h ⊗ q1h == gradedrange([q10 => 1, q11 => 1])
    @test q11 ⊗ q11 == gradedrange([q20 => 1, q21 => 1, q22 => 1])
  end

  @testset "U(1) × SU(2)" begin
    q0 = sector()
    q0h = sector(; J=SU{2}(2))
    q10 = (N=U1(1),) × (J=SU{2}(1),)
    # Put names in reverse order sometimes:
    q1h = (J=SU{2}(2),) × (N=U1(1),)
    q11 = (N=U1(1),) × (J=SU{2}(3),)
    q20 = sector(; N=U1(2))
    q2h = (N=U1(2),) × (J=SU{2}(2),)
    q21 = (N=U1(2),) × (J=SU{2}(3),)
    q22 = (N=U1(2),) × (J=SU{2}(5),)

    @test q1h ⊗ q1h == gradedrange([q20 => 1, q21 => 1])
    @test q10 ⊗ q1h == gradedrange([q2h => 1])
    @test q0h ⊗ q1h == gradedrange([q10 => 1, q11 => 1])
    @test q11 ⊗ q11 == gradedrange([q20 => 1, q21 => 1, q22 => 1])
  end

  @testset "Comparisons with unspecified labels" begin
    q2 = sector(; N=U1(2))
    q20 = (N=U1(2),) × (J=SU{2}(1),)
    @test q20 == q2

    q21 = (N=U1(2),) × (J=SU{2}(3),)
    @test q21 != q2

    a = (A=U1(0),) × (B=U1(2),)
    b = (B=U1(2),) × (C=U1(0),)
    @test a == b
    c = (B=U1(2),) × (C=U1(1),)
    @test a != c
  end
end

@testset "Test Ordered Products" begin
  @testset "Ordered Constructor" begin
    s = sector(U1(1), U1(2))
    @test length(categories(s)) == 2
    @test (@inferred quantum_dimension(s)) == 1
    @test dual(s) == sector(U1(-1), U1(-2))
    @test categories(s)[1] == U1(1)
    @test categories(s)[2] == U1(2)

    s = U1(1) × SU2(1//2) × U1(3)
    @test length(categories(s)) == 3
    @test (@inferred quantum_dimension(s)) == 2
    @test dual(s) == U1(-1) × SU2(1//2) × U1(-3)
    @test categories(s)[1] == U1(1)
    @test categories(s)[2] == SU2(1//2)
    @test categories(s)[3] == U1(3)

    s = U1(3) × SU2(1//2) × Fib("τ")
    @test length(categories(s)) == 3
    @test (@inferred quantum_dimension(s)) == 1.0 + √5
    @test dual(s) == U1(-3) × SU2(1//2) × Fib("τ")
    @test categories(s)[1] == U1(3)
    @test categories(s)[2] == SU2(1//2)
    @test categories(s)[3] == Fib("τ")
  end

  @testset "Enforce same spaces in fusion" begin
    p12 = U1(1) × U1(2)
    p123 = U1(1) × U1(2) × U1(3)
    @test_throws MethodError p12 ⊗ p123

    z12 = Z{2}(1) × Z{2}(1)
    @test_throws MethodError p12 ⊗ z12
  end

  @testset "Empty category" begin
    s = CategoryProduct(())
    @test dual(s) == s
    @test s × s == s
    @test s ⊗ s == s
    @test (@inferred quantum_dimension(s)) == 0
  end

  @testset "Fusion of Abelian products" begin
    p11 = U1(1) × U1(1)
    @test p11 ⊗ p11 == U1(2) × U1(2)

    p123 = U1(1) × U1(2) × U1(3)
    @test p123 ⊗ p123 == U1(2) × U1(4) × U1(6)

    s1 = sector(U1(1), Z{2}(1))
    s2 = sector(U1(0), Z{2}(0))
    @test s1 ⊗ s2 == U1(1) × Z{2}(1)
  end

  @testset "Fusion of NonAbelian products" begin
    phh = SU2(1//2) × SU2(1//2)
    @test phh ⊗ phh == gradedrange([
      (SU2(0) × SU2(0)) => 1,
      (SU2(1) × SU2(0)) => 1,
      (SU2(0) × SU2(1)) => 1,
      (SU2(1) × SU2(1)) => 1,
    ])
    @test (@inferred quantum_dimension(phh ⊗ phh)) == 16
  end

  @testset "Fusion of NonGroupCategory products" begin
    ı = Fib("1")
    τ = Fib("τ")
    s = ı × ı
    g = gradedrange([(ı × ı) => 1])
    @test s ⊗ s == g
    @test_broken (@inferred quantum_dimension(g)) == 1.0  # I don't understand

    s = τ × τ
    g = gradedrange([(ı × ı) => 1, (τ × ı) => 1, (ı × τ) => 1, (τ × τ) => 1])
    @test s ⊗ s == g
    @test_broken (@inferred quantum_dimension(g)) == 2.0 + 3quantum_dimension(τ)  # I don't understand

    σ = Ising("σ")
    ψ = Ising("ψ")
    s = τ × σ
    g = gradedrange([(ı × Ising(1)) => 1, (τ × Ising(1)) => 1, (ı × ψ) => 1, (τ × ψ) => 1])
    @test s ⊗ s == g
    @test (@inferred quantum_dimension(g)) == 2.0 + 2quantum_dimension(τ)  # ???
  end

  @testset "Fusion of mixed Abelian and NonAbelian products" begin
    p2h = U1(2) × SU2(1//2)
    p1h = U1(1) × SU2(1//2)
    @test p2h ⊗ p1h == gradedrange([(U1(3) × SU2(0)) => 1, (U1(3) × SU2(1)) => 1])

    p1h1 = U1(1) × SU2(1//2) × Z{2}(1)
    @test p1h1 ⊗ p1h1 ==
      gradedrange([(U1(2) × SU2(0) × Z{2}(0)) => 1, (U1(2) × SU2(1) × Z{2}(0)) => 1])
    @test (@inferred quantum_dimension(p1h1 ⊗ p1h1)) == 4
  end

  @testset "Fusion of fully mixed products" begin
    s = U1(1) × SU2(1//2) × Ising("σ")
    @test s ⊗ s == gradedrange([
      (U1(2) × SU2(0) × Ising(1)) => 1,
      (U1(2) × SU2(1) × Ising(1)) => 1,
      (U1(2) × SU2(0) × Ising("ψ")) => 1,
      (U1(2) × SU2(1) × Ising("ψ")) => 1,
    ])
    @test (@inferred quantum_dimension(s ⊗ s)) == 8

    ı = Fib("1")
    τ = Fib("τ")
    s = U1(1) × SU2(1//2) × τ
    @test s ⊗ s == gradedrange([
      (U1(2) × SU2(0) × ı) => 1,
      (U1(2) × SU2(1) × ı) => 1,
      (U1(2) × SU2(0) × τ) => 1,
      (U1(2) × SU2(1) × τ) => 1,
    ])
    @test (@inferred quantum_dimension(s ⊗ s)) == 4.0 + 4.0quantum_dimension(τ)

    s = U1(1) × ı × τ
    @test s ⊗ s == gradedrange([(U1(2) × ı × ı) => 1, (U1(2) × ı × τ) => 1])
    @test_broken (@inferred quantum_dimension(s ⊗ s)) == 1.0 + quantum_dimension(τ)
  end
end
end
