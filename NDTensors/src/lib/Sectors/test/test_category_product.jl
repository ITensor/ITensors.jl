@eval module $(gensym())
using NDTensors.Sectors:
  ×,
  ⊗,
  Fib,
  Ising,
  SU,
  SU2,
  U1,
  Z,
  block_dimensions,
  categories,
  quantum_dimension,
  sector,
  trivial
using NDTensors.GradedAxes: dual, fusion_product, gradedisequal, gradedrange
using Test: @inferred, @test, @testset, @test_throws

macro inferred_latest(ex)
  if VERSION < v"1.10"
    return esc(:($ex))
  end
  return esc(:(@inferred $ex))
end

@testset "Test Ordered Products" begin
  @testset "Ordered Constructor" begin
    s = sector(U1(1))
    @test length(categories(s)) == 1
    @test (@inferred quantum_dimension(s)) == 1
    @test (@inferred dual(s)) == sector(U1(-1))
    @test categories(s)[1] == U1(1)
    @test (@inferred_latest trivial(s)) == sector(U1(0))

    s = sector(U1(1), U1(2))
    @test length(categories(s)) == 2
    @test (@inferred quantum_dimension(s)) == 1
    @test (@inferred dual(s)) == sector(U1(-1), U1(-2))
    @test categories(s)[1] == U1(1)
    @test categories(s)[2] == U1(2)
    @test (@inferred_latest trivial(s)) == sector(U1(0), U1(0))

    s = U1(1) × SU2(1//2) × U1(3)
    @test length(categories(s)) == 3
    @test (@inferred quantum_dimension(s)) == 2
    @test (@inferred dual(s)) == U1(-1) × SU2(1//2) × U1(-3)
    @test categories(s)[1] == U1(1)
    @test categories(s)[2] == SU2(1//2)
    @test categories(s)[3] == U1(3)
    @test (@inferred_latest trivial(s)) == sector(U1(0), SU2(0), U1(0))
    @test (@inferred sector(typeof(categories(s)), categories(s))) == s
    @test (@inferred sector(typeof(s), categories(s))) == s

    s = U1(3) × SU2(1//2) × Fib("τ")
    @test length(categories(s)) == 3
    @test (@inferred quantum_dimension(s)) == 1.0 + √5
    @test dual(s) == U1(-3) × SU2(1//2) × Fib("τ")
    @test categories(s)[1] == U1(3)
    @test categories(s)[2] == SU2(1//2)
    @test categories(s)[3] == Fib("τ")
    @test (@inferred_latest trivial(s)) == sector(U1(0), SU2(0), Fib("1"))
  end

  @testset "Ordered comparisons" begin
    # convention: categories must have same length to evaluate as equal
    @test sector(U1(1), SU2(1)) == sector(U1(1), SU2(1))
    @test sector(U1(1), SU2(0)) != sector(U1(1), SU2(1))
    @test sector(U1(0), SU2(1)) != sector(U1(1), SU2(1))
    @test sector(U1(1)) != sector(U1(1), U1(0))

    # convention: categories must have same length to be compared
    @test sector(U1(0)) < sector((U1(1)))
    @test sector(U1(0), U1(2)) < sector((U1(1)), U1(0))
    @test_throws ArgumentError sector(U1(0)) < sector(U1(1), U1(2))
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
    @test (@inferred block_dimensions(g)) == [1, 3, 3, 9]

    # mixed group
    g = gradedrange([(U1(2) × SU2(0) × Z{2}(0)) => 1, (U1(2) × SU2(1) × Z{2}(0)) => 1])
    @test (@inferred quantum_dimension(g)) == 4
    @test (@inferred block_dimensions(g)) == [1, 3]
    g = gradedrange([(SU2(0) × U1(0) × SU2(1//2)) => 1, (SU2(0) × U1(1) × SU2(1//2)) => 1])
    @test (@inferred quantum_dimension(g)) == 4
    @test (@inferred block_dimensions(g)) == [2, 2]

    # NonGroupCategory
    g_fib = gradedrange([(Fib("1") × Fib("1")) => 1])
    g_ising = gradedrange([(Ising("1") × Ising("1")) => 1])
    @test (@inferred quantum_dimension((Fib("1") × Fib("1")))) == 1.0
    @test (@inferred quantum_dimension(g_fib)) == 1.0
    @test (@inferred quantum_dimension(g_ising)) == 1.0
    @test (@inferred quantum_dimension((Ising("1") × Ising("1")))) == 1.0
    @test (@inferred block_dimensions(g_fib)) == [1.0]
    @test (@inferred block_dimensions(g_ising)) == [1.0]

    @test (@inferred quantum_dimension(U1(1) × Fib("1"))) == 1.0
    @test (@inferred quantum_dimension(gradedrange([U1(1) × Fib("1") => 1]))) == 1.0

    # mixed product Abelian / NonAbelian / NonGroup
    g = gradedrange([
      (U1(2) × SU2(0) × Ising(1)) => 1,
      (U1(2) × SU2(1) × Ising(1)) => 1,
      (U1(2) × SU2(0) × Ising("ψ")) => 1,
      (U1(2) × SU2(1) × Ising("ψ")) => 1,
    ])
    @test (@inferred quantum_dimension(g)) == 8.0
    @test (@inferred block_dimensions(g)) == [1.0, 3.0, 1.0, 3.0]

    ϕ = (1 + √5) / 2
    g = gradedrange([
      (Fib("1") × SU2(0) × U1(2)) => 1,
      (Fib("1") × SU2(1) × U1(2)) => 1,
      (Fib("τ") × SU2(0) × U1(2)) => 1,
      (Fib("τ") × SU2(1) × U1(2)) => 1,
    ])
    @test (@inferred quantum_dimension(g)) == 4.0 + 4.0ϕ
    @test (@inferred block_dimensions(g)) == [1.0, 3.0, 1.0ϕ, 3.0ϕ]
  end

  @testset "Fusion of Abelian products" begin
    p1 = sector(U1(1))
    p2 = sector(U1(2))
    @test (@inferred p1 ⊗ p2) == sector(U1(3))

    p11 = U1(1) × U1(1)
    @test (@inferred p11 ⊗ p11) == U1(2) × U1(2)

    p123 = U1(1) × U1(2) × U1(3)
    @test (@inferred p123 ⊗ p123) == U1(2) × U1(4) × U1(6)

    s1 = sector(U1(1), Z{2}(1))
    s2 = sector(U1(0), Z{2}(0))
    @test (@inferred s1 ⊗ s2) == U1(1) × Z{2}(1)
  end

  @testset "Fusion of NonAbelian products" begin
    p0 = sector(SU2(0))
    ph = sector(SU2(1//2))
    @test gradedisequal((@inferred p0 ⊗ ph), gradedrange([sector(SU2(1//2)) => 1]))

    phh = SU2(1//2) × SU2(1//2)
    @test gradedisequal(
      phh ⊗ phh,
      gradedrange([
        (SU2(0) × SU2(0)) => 1,
        (SU2(1) × SU2(0)) => 1,
        (SU2(0) × SU2(1)) => 1,
        (SU2(1) × SU2(1)) => 1,
      ]),
    )
    @test gradedisequal(
      (@inferred phh ⊗ phh),
      gradedrange([
        (SU2(0) × SU2(0)) => 1,
        (SU2(1) × SU2(0)) => 1,
        (SU2(0) × SU2(1)) => 1,
        (SU2(1) × SU2(1)) => 1,
      ]),
    )
  end

  @testset "Fusion of NonGroupCategory products" begin
    ı = Fib("1")
    τ = Fib("τ")
    s = ı × ı
    @test gradedisequal((@inferred s ⊗ s), gradedrange([s => 1]))

    s = τ × τ
    @test gradedisequal(
      (@inferred s ⊗ s),
      gradedrange([(ı × ı) => 1, (τ × ı) => 1, (ı × τ) => 1, (τ × τ) => 1]),
    )

    σ = Ising("σ")
    ψ = Ising("ψ")
    s = τ × σ
    g = gradedrange([
      (ı × Ising("1")) => 1, (τ × Ising("1")) => 1, (ı × ψ) => 1, (τ × ψ) => 1
    ])
    @test gradedisequal((@inferred s ⊗ s), g)
  end

  @testset "Fusion of mixed Abelian and NonAbelian products" begin
    p2h = U1(2) × SU2(1//2)
    p1h = U1(1) × SU2(1//2)
    @test gradedisequal(
      (@inferred p2h ⊗ p1h), gradedrange([(U1(3) × SU2(0)) => 1, (U1(3) × SU2(1)) => 1])
    )

    p1h1 = U1(1) × SU2(1//2) × Z{2}(1)
    @test gradedisequal(
      (@inferred p1h1 ⊗ p1h1),
      gradedrange([(U1(2) × SU2(0) × Z{2}(0)) => 1, (U1(2) × SU2(1) × Z{2}(0)) => 1]),
    )
  end

  @testset "Fusion of fully mixed products" begin
    s = U1(1) × SU2(1//2) × Ising("σ")
    @test gradedisequal(
      (@inferred s ⊗ s),
      gradedrange([
        (U1(2) × SU2(0) × Ising("1")) => 1,
        (U1(2) × SU2(1) × Ising("1")) => 1,
        (U1(2) × SU2(0) × Ising("ψ")) => 1,
        (U1(2) × SU2(1) × Ising("ψ")) => 1,
      ]),
    )

    ı = Fib("1")
    τ = Fib("τ")
    s = SU2(1//2) × U1(1) × τ
    @test gradedisequal(
      (@inferred s ⊗ s),
      gradedrange([
        (SU2(0) × U1(2) × ı) => 1,
        (SU2(1) × U1(2) × ı) => 1,
        (SU2(0) × U1(2) × τ) => 1,
        (SU2(1) × U1(2) × τ) => 1,
      ]),
    )

    s = U1(1) × ı × τ
    @test gradedisequal(
      (@inferred s ⊗ s), gradedrange([(U1(2) × ı × ı) => 1, (U1(2) × ı × τ) => 1])
    )
  end

  @testset "Fusion of different length Categories" begin
    @test sector(U1(1) × U1(0)) ⊗ sector(U1(1)) == sector(U1(2) × U1(0))
    @test gradedisequal(
      (@inferred sector(SU2(0) × SU2(0)) ⊗ sector(SU2(1))),
      gradedrange([sector(SU2(1) × SU2(0)) => 1]),
    )

    @test gradedisequal(
      (@inferred sector(SU2(1) × U1(1)) ⊗ sector(SU2(0))),
      gradedrange([sector(SU2(1) × U1(1)) => 1]),
    )
    @test gradedisequal(
      (@inferred sector(U1(1) × SU2(1)) ⊗ sector(U1(2))),
      gradedrange([sector(U1(3) × SU2(1)) => 1]),
    )

    # check incompatible categories
    p12 = Z{2}(1) × U1(2)
    z12 = Z{2}(1) × Z{2}(1)
    @test_throws MethodError p12 ⊗ z12
  end

  @testset "GradedUnitRange fusion rules" begin
    s1 = U1(1) × SU2(1//2) × Ising("σ")
    s2 = U1(0) × SU2(1//2) × Ising("1")
    g1 = gradedrange([s1 => 2])
    g2 = gradedrange([s2 => 1])
    @test gradedisequal(
      (@inferred fusion_product(g1, g2)),
      gradedrange([U1(1) × SU2(0) × Ising("σ") => 2, U1(1) × SU2(1) × Ising("σ") => 2]),
    )
  end
end

@testset "Test Named Category Products" begin
  @testset "Construct from × of NamedTuples" begin
    s = (A=U1(1),) × (B=Z{2}(0),)
    @test length(categories(s)) == 2
    @test categories(s)[:A] == U1(1)
    @test categories(s)[:B] == Z{2}(0)
    @test (@inferred quantum_dimension(s)) == 1
    @test (@inferred dual(s)) == (A=U1(-1),) × (B=Z{2}(0),)
    @test (@inferred_latest trivial(s)) == (A=U1(0),) × (B=Z{2}(0),)

    s = (A=U1(1),) × (B=SU2(2),)
    @test length(categories(s)) == 2
    @test categories(s)[:A] == U1(1)
    @test categories(s)[:B] == SU2(2)
    @test (@inferred quantum_dimension(s)) == 5
    @test (@inferred dual(s)) == (A=U1(-1),) × (B=SU2(2),)
    @test (@inferred_latest trivial(s)) == (A=U1(0),) × (B=SU2(0),)
    @test (@inferred sector(typeof(categories(s)), Tuple(categories(s)))) == s
    @test (@inferred sector(typeof(s), Tuple(categories(s)))) == s

    s = s × (C=Ising("ψ"),)
    @test length(categories(s)) == 3
    @test categories(s)[:C] == Ising("ψ")
    @test (@inferred_latest quantum_dimension(s)) == 5.0
    @test (@inferred dual(s)) == (A=U1(-1),) × (B=SU2(2),) × (C=Ising("ψ"),)

    s1 = (A=U1(1),) × (B=Z{2}(0),)
    s2 = (A=U1(1),) × (C=Z{2}(0),)
    @test_throws ArgumentError s1 × s2
  end

  @testset "Construct from Pairs" begin
    s = sector("A" => U1(2))
    @test length(categories(s)) == 1
    @test categories(s)[:A] == U1(2)
    @test s == sector(; A=U1(2))
    @test (@inferred quantum_dimension(s)) == 1
    @test (@inferred dual(s)) == sector("A" => U1(-2))
    @test (@inferred_latest trivial(s)) == sector(; A=U1(0))

    s = sector("B" => Ising("ψ"), :C => Z{2}(1))
    @test length(categories(s)) == 2
    @test categories(s)[:B] == Ising("ψ")
    @test categories(s)[:C] == Z{2}(1)
    @test (@inferred quantum_dimension(s)) == 1.0
  end

  @testset "Comparisons with unspecified labels" begin
    # convention: categories evaluate as equal if unmatched labels are trivial
    # this is different from ordered tuple convention
    q2 = sector(; N=U1(2))
    q20 = (N=U1(2),) × (J=SU2(0),)
    @test q20 == q2
    @test !(q20 < q2)
    @test !(q2 < q20)

    q21 = (N=U1(2),) × (J=SU2(1),)
    @test q21 != q2
    @test q20 < q21
    @test q2 < q21

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
    @test (@inferred quantum_dimension(g)) == 4
    g = gradedrange([
      sector(; A=SU2(0), B=Z{2}(0), C=SU2(1//2)) => 1,
      sector(; A=SU2(0), B=Z{2}(1), C=SU2(1//2)) => 1,
    ])
    @test (@inferred quantum_dimension(g)) == 4

    # non group categories
    g_fib = gradedrange([sector(; A=Fib("1"), B=Fib("1")) => 1])
    g_ising = gradedrange([sector(; A=Ising("1"), B=Ising("1")) => 1])
    @test (@inferred quantum_dimension(g_fib)) == 1.0
    @test (@inferred quantum_dimension(g_ising)) == 1.0

    # mixed product Abelian / NonAbelian / NonGroup
    g = gradedrange([
      sector(; A=U1(2), B=SU2(0), C=Ising(1)) => 1,
      sector(; A=U1(2), B=SU2(1), C=Ising(1)) => 1,
      sector(; A=U1(2), B=SU2(0), C=Ising("ψ")) => 1,
      sector(; A=U1(2), B=SU2(1), C=Ising("ψ")) => 1,
    ])
    @test (@inferred_latest quantum_dimension(g)) == 8.0

    g = gradedrange([
      sector(; A=Fib("1"), B=SU2(0), C=U1(2)) => 1,
      sector(; A=Fib("1"), B=SU2(1), C=U1(2)) => 1,
      sector(; A=Fib("τ"), B=SU2(0), C=U1(2)) => 1,
      sector(; A=Fib("τ"), B=SU2(1), C=U1(2)) => 1,
    ])
    @test (@inferred quantum_dimension(g)) == 4.0 + 4.0quantum_dimension(Fib("τ"))
  end

  @testset "Fusion of Abelian products" begin
    q00 = sector()
    q10 = sector(; A=U1(1))
    q01 = sector(; B=U1(1))
    q11 = sector(; A=U1(1), B=U1(1))

    @test (@inferred q10 ⊗ q10) == sector(; A=U1(2))
    @test (@inferred q01 ⊗ q00) == q01
    @test (@inferred q00 ⊗ q01) == q01
    @test (@inferred q10 ⊗ q01) == q11
    @test (@inferred q11 ⊗ q11) == sector(; A=U1(2), B=U1(2))

    s11 = sector(; A=U1(1), B=Z{2}(1))
    s10 = sector(; A=U1(1))
    s01 = sector(; B=Z{2}(1))
    @test (@inferred s01 ⊗ q00) == s01
    @test (@inferred q00 ⊗ s01) == s01
    @test (@inferred s10 ⊗ s01) == s11
    @test (@inferred s11 ⊗ s11) == sector(; A=U1(2), B=Z{2}(0))
  end

  @testset "Fusion of NonAbelian products" begin
    p0 = sector()
    pha = sector(; A=SU2(1//2))
    phb = sector(; B=SU2(1//2))
    phab = sector(; A=SU2(1//2), B=SU2(1//2))

    @test gradedisequal(
      (@inferred pha ⊗ pha), gradedrange([sector(; A=SU2(0)) => 1, sector(; A=SU2(1)) => 1])
    )
    @test gradedisequal((@inferred pha ⊗ p0), gradedrange([pha => 1]))
    @test gradedisequal((@inferred p0 ⊗ phb), gradedrange([phb => 1]))
    @test gradedisequal((@inferred pha ⊗ phb), gradedrange([phab => 1]))

    @test gradedisequal(
      (@inferred phab ⊗ phab),
      gradedrange([
        sector(; A=SU2(0), B=SU2(0)) => 1,
        sector(; A=SU2(1), B=SU2(0)) => 1,
        sector(; A=SU2(0), B=SU2(1)) => 1,
        sector(; A=SU2(1), B=SU2(1)) => 1,
      ]),
    )
  end

  @testset "Fusion of NonGroupCategory products" begin
    ı = Fib("1")
    τ = Fib("τ")
    s = sector(; A=ı, B=ı)
    @test gradedisequal((@inferred s ⊗ s), gradedrange([s => 1]))

    s = sector(; A=τ, B=τ)
    @test gradedisequal(
      (@inferred s ⊗ s),
      gradedrange([
        sector(; A=ı, B=ı) => 1,
        sector(; A=τ, B=ı) => 1,
        sector(; A=ı, B=τ) => 1,
        sector(; A=τ, B=τ) => 1,
      ]),
    )

    σ = Ising("σ")
    ψ = Ising("ψ")
    s = sector(; A=τ, B=σ)
    g = gradedrange([
      sector(; A=ı, B=Ising("1")) => 1,
      sector(; A=τ, B=Ising("1")) => 1,
      sector(; A=ı, B=ψ) => 1,
      sector(; A=τ, B=ψ) => 1,
    ])
    @test gradedisequal((@inferred s ⊗ s), g)
  end

  @testset "Fusion of mixed Abelian and NonAbelian products" begin
    q0h = sector(; J=SU2(1//2))
    q10 = (N=U1(1),) × (J=SU2(0),)
    # Put names in reverse order sometimes:
    q1h = (J=SU2(1//2),) × (N=U1(1),)
    q11 = (N=U1(1),) × (J=SU2(1),)
    q20 = (N=U1(2),) × (J=SU2(0),)  # julia 1.6 does not accept gradedrange without J
    q2h = (N=U1(2),) × (J=SU2(1//2),)
    q21 = (N=U1(2),) × (J=SU2(1),)
    q22 = (N=U1(2),) × (J=SU2(2),)

    @test gradedisequal((@inferred q1h ⊗ q1h), gradedrange([q20 => 1, q21 => 1]))
    @test gradedisequal((@inferred q10 ⊗ q1h), gradedrange([q2h => 1]))
    @test gradedisequal((@inferred q0h ⊗ q1h), gradedrange([q10 => 1, q11 => 1]))
    @test gradedisequal((@inferred q11 ⊗ q11), gradedrange([q20 => 1, q21 => 1, q22 => 1]))
  end

  @testset "Fusion of fully mixed products" begin
    s = sector(; A=U1(1), B=SU2(1//2), C=Ising("σ"))
    @test gradedisequal(
      (@inferred s ⊗ s),
      gradedrange([
        sector(; A=U1(2), B=SU2(0), C=Ising("1")) => 1,
        sector(; A=U1(2), B=SU2(1), C=Ising("1")) => 1,
        sector(; A=U1(2), B=SU2(0), C=Ising("ψ")) => 1,
        sector(; A=U1(2), B=SU2(1), C=Ising("ψ")) => 1,
      ]),
    )

    ı = Fib("1")
    τ = Fib("τ")
    s = sector(; A=SU2(1//2), B=U1(1), C=τ)
    @test gradedisequal(
      (@inferred s ⊗ s),
      gradedrange([
        sector(; A=SU2(0), B=U1(2), C=ı) => 1,
        sector(; A=SU2(1), B=U1(2), C=ı) => 1,
        sector(; A=SU2(0), B=U1(2), C=τ) => 1,
        sector(; A=SU2(1), B=U1(2), C=τ) => 1,
      ]),
    )

    s = sector(; A=τ, B=U1(1), C=ı)
    @test gradedisequal(
      (@inferred s ⊗ s),
      gradedrange([sector(; B=U1(2), A=ı, C=ı) => 1, sector(; B=U1(2), A=τ, C=ı) => 1]),
    )
  end
  @testset "GradedUnitRange fusion rules" begin
    s1 = sector(; A=U1(1), B=SU2(1//2), C=Ising("σ"))
    s2 = sector(; A=U1(0), B=SU2(1//2), C=Ising("1"))
    g1 = gradedrange([s1 => 2])
    g2 = gradedrange([s2 => 1])
    s3 = sector(; A=U1(1), B=SU2(0), C=Ising("σ"))
    s4 = sector(; A=U1(1), B=SU2(1), C=Ising("σ"))
    @test gradedisequal(
      (@inferred_latest fusion_product(g1, g2)), gradedrange([s3 => 2, s4 => 2])
    )

    sA = sector(; A=U1(1))
    sB = sector(; B=SU2(1//2))
    sAB = sector(; A=U1(1), B=SU2(1//2))
    gA = gradedrange([sA => 2])
    gB = gradedrange([sB => 1])
    @test gradedisequal((@inferred fusion_product(gA, gB)), gradedrange([sAB => 2]))
  end
end

@testset "Empty category" begin
  s = sector()
  @test s == s
  @test (@inferred dual(s)) == s
  @test (@inferred s × s) == s
  @test (@inferred s ⊗ s) == s
  @test (@inferred quantum_dimension(s)) == 1
  @test (@inferred_latest trivial(s)) == s
  @test typeof(s) == typeof(sector(()))
  @test typeof(s) == typeof(sector((;)))  # empty NamedTuple is cast to Tuple{}

  g0 = gradedrange([s => 2])
  @test gradedisequal((@inferred fusion_product(g0, g0)), gradedrange([s => 4]))

  @test (@inferred s × U1(1)) == sector(U1(1))
  @test (@inferred s × sector(U1(1))) == sector(U1(1))
  @test (@inferred s × sector(; A=U1(1))) == sector(; A=U1(1))
  @test (@inferred U1(1) × s) == sector(U1(1))
  @test (@inferred sector(U1(1)) × s) == sector(U1(1))
  @test (@inferred sector(; A=U1(1)) × s) == sector(; A=U1(1))

  # Empty acts as trivial
  @test (@inferred U1(1) ⊗ s) == U1(1)
  @test (@inferred SU2(0) ⊗ s) == gradedrange([SU2(0) => 1])
  @test (@inferred Fib("τ") ⊗ s) == gradedrange([Fib("τ") => 1])
  @test (@inferred s ⊗ U1(1)) == U1(1)
  @test (@inferred s ⊗ SU2(0)) == gradedrange([SU2(0) => 1])
  @test (@inferred s ⊗ Fib("τ")) == gradedrange([Fib("τ") => 1])

  @test (@inferred sector(U1(1)) ⊗ s) == sector(U1(1))
  @test (@inferred sector(SU2(0)) ⊗ s) == gradedrange([sector(SU2(0)) => 1])
  @test (@inferred sector(Fib("τ"), SU2(1), U1(2)) ⊗ s) ==
    gradedrange([sector(Fib("τ"), SU2(1), U1(2)) => 1])

  @test (@inferred sector(; A=U1(1)) ⊗ s) == sector(; A=U1(1))
  @test (@inferred sector(; A=SU2(0)) ⊗ s) == gradedrange([sector(; A=SU2(0)) => 1])
  @test (@inferred sector(; A=Fib("τ"), B=SU2(1), C=U1(2)) ⊗ s) ==
    gradedrange([sector(; A=Fib("τ"), B=SU2(1), C=U1(2)) => 1])

  # Empty behaves as empty NamedTuple
  @test s != U1(0)
  @test s != sector(U1(0))
  @test s != sector(; A=U1(1))
  @test s == sector(; A=U1(0))
  @test sector(; A=U1(0)) == s

  @test !(s < s)
  @test_throws ArgumentError s < sector(U1(0))
  @test s < sector(; A=U1(1))
  @test s > sector(; A=U1(-1))
  @test !(s < sector(; A=U1(0)))
  @test !(s > sector(; A=U1(0)))
end
end
