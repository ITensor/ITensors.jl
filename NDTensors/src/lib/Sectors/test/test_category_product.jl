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

  @testset "Quantum dimension and GradedUnitRange" begin
    g = gradedrange([sector(; A=U1(0), B=Z{2}(0)) => 1, sector(; A=U1(1), B=Z{2}(0)) => 2])  # abelian
    @test (@inferred quantum_dimension(g)) == 3

    g = gradedrange([  # non-abelian
      sector(; A=SU2(0), B=SU2(0)) => 1,
      sector(; A=SU2(1), B=SU2(0)) => 1,
      sector(; A=SU2(0), B=SU2(1)) => 1,
      sector(; A=SU2(1), B=SU2(1)) => 1,
    ])
    @test (@inferred quantum_dimension(g)) == 16

    # mixed group
    g = gradedrange([
      sector(; A=U1(2), B=SU2(0), C=Z{2}(0)) => 1,
      sector(; A=U1(2), B=SU2(1), C=Z{2}(0)) => 1,
    ])
    @test_broken (@inferred quantum_dimension(g)) == 4  # TODO

    # non group categories
    # make no sense, see Ordered Products
    g_fib = gradedrange([sector(; A=Fib("1"), B=Fib("1")) => 1])
    g_ising = gradedrange([sector(; A=Ising("1"), B=Ising("1")) => 1])
    @test_broken (@inferred quantum_dimension(g_fib)) == 1.0
    @test (@inferred quantum_dimension(g_ising)) == 1.0

    # mixed product Abelian / NonAbelian / NonGroup
    g = gradedrange([
      sector(; A=U1(2), B=SU2(0), C=Ising(1)) => 1,
      sector(; A=U1(2), B=SU2(1), C=Ising(1)) => 1,
      sector(; A=U1(2), B=SU2(0), C=Ising("ψ")) => 1,
      sector(; A=U1(2), B=SU2(1), C=Ising("ψ")) => 1,
    ])
    @test @inferred(quantum_dimension(g)) == 8.0

    g = gradedrange([
      sector(; A=U1(2), B=SU2(0), C=Fib("1")) => 1,
      sector(; A=U1(2), B=SU2(1), C=Fib("1")) => 1,
      sector(; A=U1(2), B=SU2(0), C=Fib("τ")) => 1,
      sector(; A=U1(2), B=SU2(1), C=Fib("τ")) => 1,
    ])
    @test (@inferred quantum_dimension(g)) == 4.0 + 4.0quantum_dimension(Fib("τ"))
  end

  @testset "Empty category" begin
    s = sector()
    @test @inferred(dual(s)) == s
    @test @inferred(s × s) == s
    @test @inferred(s ⊗ s) == s
    @test (@inferred quantum_dimension(s)) == 0
  end

  @testset "Fusion of Abelian products" begin
    q00 = sector()
    q10 = sector(; A=U1(1))
    q01 = sector(; B=U1(1))
    q11 = sector(; A=U1(1), B=U1(1))

    @test @inferred(q10 ⊗ q10) == sector(; A=U1(2))
    @test @inferred(q01 ⊗ q00) == q01
    @test @inferred(q00 ⊗ q01) == q01
    @test @inferred(q10 ⊗ q01) == q11
    @test @inferred(q11 ⊗ q11) == sector(; A=U1(2), B=U1(2))

    s11 = sector(; A=U1(1), B=Z{2}(1))
    s10 = sector(; A=U1(1))
    s01 = sector(; B=Z{2}(1))
    @test @inferred(s01 ⊗ q00) == s01
    @test @inferred(q00 ⊗ s01) == s01
    @test @inferred(s10 ⊗ s01) == s11
    @test @inferred(s11 ⊗ s11) == sector(; A=U1(2), B=Z{2}(0))
  end

  @testset "Fusion of NonAbelian products" begin
    p0 = sector()
    pha = sector(; A=SU2(1//2))
    phb = sector(; B=SU2(1//2))
    phab = sector(; A=SU2(1//2), B=SU2(1//2))

    @test (@inferred pha ⊗ pha) ==
      gradedrange([sector(; A=SU2(0)) => 1, sector(; A=SU2(1)) => 1])
    @test (@inferred pha ⊗ p0) == gradedrange([pha => 1])
    @test (@inferred p0 ⊗ phb) == gradedrange([phb => 1])
    @test (@inferred pha ⊗ phb) == gradedrange([phab => 1])

    @test (@inferred phab ⊗ phab) == gradedrange([
      sector(; A=SU2(0), B=SU2(0)) => 1,
      sector(; A=SU2(1), B=SU2(0)) => 1,
      sector(; A=SU2(0), B=SU2(1)) => 1,
      sector(; A=SU2(1), B=SU2(1)) => 1,
    ])
  end

  @testset "Fusion of NonGroupCategory products" begin
    ı = Fib("1")
    τ = Fib("τ")
    s = sector(; A=ı, B=ı)
    @test_broken @inferred(s ⊗ s) == gradedrange([s => 1])  # TODO

    s = sector(; A=τ, B=τ)
    @test @inferred(s ⊗ s) == gradedrange([
      sector(; A=ı, B=ı) => 1,
      sector(; A=τ, B=ı) => 1,
      sector(; A=ı, B=τ) => 1,
      sector(; A=τ, B=τ) => 1,
    ])

    σ = Ising("σ")
    ψ = Ising("ψ")
    s = τ × σ
    g = gradedrange([
      sector(; A=ı, B=Ising(1)) => 1,
      sector(; A=τ, B=Ising(1)) => 1,
      sector(; A=ı, B=ψ) => 1,
      sector(; A=τ, B=ψ) => 1,
    ])
    @test @inferred(s ⊗ s) == g
  end

  @testset "Fusion of mixed Abelian and NonAbelian products" begin
    q0h = sector(; J=SU2(1//2))
    q10 = (N=U1(1),) × (J=SU2(0),)
    # Put names in reverse order sometimes:
    q1h = (J=SU2(1//2),) × (N=U1(1),)
    q11 = (N=U1(1),) × (J=SU2(1),)
    q20 = sector(; N=U1(2))
    q2h = (N=U1(2),) × (J=SU2(1//2),)
    q21 = (N=U1(2),) × (J=SU2(1),)
    q22 = (N=U1(2),) × (J=SU2(2),)

    @test @inferred(q1h ⊗ q1h) == gradedrange([q20 => 1, q21 => 1])
    @test_broken @inferred(q10 ⊗ q1h) == gradedrange([q2h => 1])  # TODO
    @test @inferred(q0h ⊗ q1h) == gradedrange([q10 => 1, q11 => 1])
    @test @inferred(q11 ⊗ q11) == gradedrange([q20 => 1, q21 => 1, q22 => 1])
  end

  @testset "Fusion of fully mixed products" begin
    s = sector(; A=U1(1), B=SU2(1//2), C=Ising("σ"))
    @test @inferred(s ⊗ s) == gradedrange([
      sector(; A=U1(2), B=SU2(0), C=Ising(1)) => 1,
      sector(; A=U1(2), B=SU2(1), C=Ising(1)) => 1,
      sector(; A=U1(2), B=SU2(0), C=Ising("ψ")) => 1,
      sector(; A=U1(2), B=SU2(1), C=Ising("ψ")) => 1,
    ])

    ı = Fib("1")
    τ = Fib("τ")
    s = U1(1) × SU2(1//2) × τ
    @test_broken @inferred(s ⊗ s) == gradedrange([  # TODO
      sector(; A=U1(2), B=SU2(0), C=ı) => 1,
      sector(; A=U1(2), B=SU2(1), C=ı) => 1,
      sector(; A=U1(2), B=SU2(0), C=τ) => 1,
      sector(; A=U1(2), B=SU2(1), C=τ) => 1,
    ])

    s = U1(1) × ı × τ
    @test @inferred(s ⊗ s) ==
      gradedrange([sector(; A=U1(2), B=ı, C=ı) => 1, sector(; A=U1(2), B=ı, C=τ) => 1])
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

  @testset "Quantum dimension and GradedUnitRange" begin
    g = gradedrange([(U1(0) × Z{2}(0)) => 1, (U1(1) × Z{2}(0)) => 2])  # abelian
    @test (@inferred quantum_dimension(g)) == 3

    g = gradedrange([  # non-abelian
      (SU2(0) × SU2(0)) => 1,
      (SU2(1) × SU2(0)) => 1,
      (SU2(0) × SU2(1)) => 1,
      (SU2(1) × SU2(1)) => 1,
    ])
    @test (@inferred quantum_dimension(g)) == 16

    # mixed group
    g = gradedrange([(U1(2) × SU2(0) × Z{2}(0)) => 1, (U1(2) × SU2(1) × Z{2}(0)) => 1])
    @test (@inferred quantum_dimension(g)) == 4

    # NonGroupCategory is strange
    g_fib = gradedrange([(Fib("1") × Fib("1")) => 1])
    g_ising = gradedrange([(Ising("1") × Ising("1")) => 1])
    # for the next 2 tests, the first one will be broken, the second will pass
    # it does not matter which one is Fib and which one is Ising
    # only compilation order matters
    # I don't understand.
    @test_broken (@inferred quantum_dimension(g_fib)) == 1.0
    @test (@inferred quantum_dimension(g_ising)) == 1.0

    # check commenting the two tests above and uncommenting the two below
    #@test_broken (@inferred quantum_dimension(g_ising)) == 1.0
    #@test (@inferred quantum_dimension(g_fib)) ==  1.0

    # or even executing the sector-wise test below *before* magically makes the tests pass
    @test (@inferred quantum_dimension((Fib("1") × Fib("1")))) == 1.0
    @test (@inferred quantum_dimension((Ising("1") × Ising("1")))) == 1.0

    # similar story below: swapping the two tests make both pass.
    @test_broken (@inferred quantum_dimension(gradedrange([U1(1) × Fib("1") => 1]))) == 1.0
    @test (@inferred quantum_dimension(U1(1) × Fib("1"))) == 1.0
    # check commenting above and uncommenting below!
    # @test (@inferred quantum_dimension(U1(1) × Fib("1"))) == 1.0
    # @test (@inferred quantum_dimension(gradedrange([U1(1) × Fib("1") => 1]))) == 1.0

    # mixed product Abelian / NonAbelian / NonGroup
    g = gradedrange([
      (U1(2) × SU2(0) × Ising(1)) => 1,
      (U1(2) × SU2(1) × Ising(1)) => 1,
      (U1(2) × SU2(0) × Ising("ψ")) => 1,
      (U1(2) × SU2(1) × Ising("ψ")) => 1,
    ])
    @test @inferred(quantum_dimension(g)) == 8.0

    g = gradedrange([
      (U1(2) × SU2(0) × Fib("1")) => 1,
      (U1(2) × SU2(1) × Fib("1")) => 1,
      (U1(2) × SU2(0) × Fib("τ")) => 1,
      (U1(2) × SU2(1) × Fib("τ")) => 1,
    ])
    @test (@inferred quantum_dimension(g)) == 4.0 + 4.0quantum_dimension(Fib("τ"))
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
    @test (@inferred dual(s)) == s
    @test (@inferred s × s) == s
    @test (@inferred s ⊗ s) == s
    @test (@inferred quantum_dimension(s)) == 0
  end

  @testset "Fusion of Abelian products" begin
    p11 = U1(1) × U1(1)
    @test @inferred(p11 ⊗ p11 == U1(2) × U1(2))

    p123 = U1(1) × U1(2) × U1(3)
    @test @inferred(p123 ⊗ p123 == U1(2) × U1(4) × U1(6))

    s1 = sector(U1(1), Z{2}(1))
    s2 = sector(U1(0), Z{2}(0))
    @test @inferred(s1 ⊗ s2 == U1(1) × Z{2}(1))
  end

  @testset "Fusion of NonAbelian products" begin
    phh = SU2(1//2) × SU2(1//2)
    @test phh ⊗ phh == gradedrange([
      (SU2(0) × SU2(0)) => 1,
      (SU2(1) × SU2(0)) => 1,
      (SU2(0) × SU2(1)) => 1,
      (SU2(1) × SU2(1)) => 1,
    ])
    @test (@inferred phh ⊗ phh == gradedrange([
      (SU2(0) × SU2(0)) => 1,
      (SU2(1) × SU2(0)) => 1,
      (SU2(0) × SU2(1)) => 1,
      (SU2(1) × SU2(1)) => 1,
    ]))
  end

  @testset "Fusion of NonGroupCategory products" begin
    ı = Fib("1")
    τ = Fib("τ")
    s = ı × ı
    @test @inferred(s ⊗ s == gradedrange([s => 1]))

    s = τ × τ
    @test @inferred(
      s ⊗ s == gradedrange([(ı × ı) => 1, (τ × ı) => 1, (ı × τ) => 1, (τ × τ) => 1])
    )

    σ = Ising("σ")
    ψ = Ising("ψ")
    s = τ × σ
    g = gradedrange([(ı × Ising(1)) => 1, (τ × Ising(1)) => 1, (ı × ψ) => 1, (τ × ψ) => 1])
    @test @inferred(s ⊗ s) == g
  end

  @testset "Fusion of mixed Abelian and NonAbelian products" begin
    p2h = U1(2) × SU2(1//2)
    p1h = U1(1) × SU2(1//2)
    @test p2h ⊗ p1h == gradedrange([(U1(3) × SU2(0)) => 1, (U1(3) × SU2(1)) => 1])

    p1h1 = U1(1) × SU2(1//2) × Z{2}(1)
    @test p1h1 ⊗ p1h1 ==
      gradedrange([(U1(2) × SU2(0) × Z{2}(0)) => 1, (U1(2) × SU2(1) × Z{2}(0)) => 1])
  end

  @testset "Fusion of fully mixed products" begin
    s = U1(1) × SU2(1//2) × Ising("σ")
    @test s ⊗ s == gradedrange([
      (U1(2) × SU2(0) × Ising(1)) => 1,
      (U1(2) × SU2(1) × Ising(1)) => 1,
      (U1(2) × SU2(0) × Ising("ψ")) => 1,
      (U1(2) × SU2(1) × Ising("ψ")) => 1,
    ])

    ı = Fib("1")
    τ = Fib("τ")
    s = U1(1) × SU2(1//2) × τ
    @test s ⊗ s == gradedrange([
      (U1(2) × SU2(0) × ı) => 1,
      (U1(2) × SU2(1) × ı) => 1,
      (U1(2) × SU2(0) × τ) => 1,
      (U1(2) × SU2(1) × τ) => 1,
    ])

    s = U1(1) × ı × τ
    @test @inferred(s ⊗ s) == gradedrange([(U1(2) × ı × ı) => 1, (U1(2) × ı × τ) => 1])
  end
end
end
