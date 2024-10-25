@eval module $(gensym())
using NDTensors.SymmetrySectors:
  ×,
  ⊗,
  Fib,
  Ising,
  SectorProduct,
  SU,
  SU2,
  TrivialSector,
  U1,
  Z,
  block_dimensions,
  quantum_dimension,
  arguments,
  trivial
using NDTensors.GradedAxes: dual, fusion_product, space_isequal, gradedrange
using Test: @inferred, @test, @testset, @test_throws

@testset "Test Ordered Products" begin
  @testset "Ordered Constructor" begin
    s = SectorProduct(U1(1))
    @test length(arguments(s)) == 1
    @test (@inferred quantum_dimension(s)) == 1
    @test (@inferred dual(s)) == SectorProduct(U1(-1))
    @test arguments(s)[1] == U1(1)
    @test (@inferred trivial(s)) == SectorProduct(U1(0))

    s = SectorProduct(U1(1), U1(2))
    @test length(arguments(s)) == 2
    @test (@inferred quantum_dimension(s)) == 1
    @test (@inferred dual(s)) == SectorProduct(U1(-1), U1(-2))
    @test arguments(s)[1] == U1(1)
    @test arguments(s)[2] == U1(2)
    @test (@inferred trivial(s)) == SectorProduct(U1(0), U1(0))

    s = U1(1) × SU2(1//2) × U1(3)
    @test length(arguments(s)) == 3
    @test (@inferred quantum_dimension(s)) == 2
    @test (@inferred dual(s)) == U1(-1) × SU2(1//2) × U1(-3)
    @test arguments(s)[1] == U1(1)
    @test arguments(s)[2] == SU2(1//2)
    @test arguments(s)[3] == U1(3)
    @test (@inferred trivial(s)) == SectorProduct(U1(0), SU2(0), U1(0))

    s = U1(3) × SU2(1//2) × Fib("τ")
    @test length(arguments(s)) == 3
    @test (@inferred quantum_dimension(s)) == 1.0 + √5
    @test dual(s) == U1(-3) × SU2(1//2) × Fib("τ")
    @test arguments(s)[1] == U1(3)
    @test arguments(s)[2] == SU2(1//2)
    @test arguments(s)[3] == Fib("τ")
    @test (@inferred trivial(s)) == SectorProduct(U1(0), SU2(0), Fib("1"))

    s = TrivialSector() × U1(3) × SU2(1 / 2)
    @test length(arguments(s)) == 3
    @test (@inferred quantum_dimension(s)) == 2
    @test dual(s) == TrivialSector() × U1(-3) × SU2(1//2)
    @test (@inferred trivial(s)) == SectorProduct(TrivialSector(), U1(0), SU2(0))
    @test s > trivial(s)
  end

  @testset "Ordered comparisons" begin
    # convention: missing arguments are filled with singlets
    @test SectorProduct(U1(1), SU2(1)) == SectorProduct(U1(1), SU2(1))
    @test SectorProduct(U1(1), SU2(0)) != SectorProduct(U1(1), SU2(1))
    @test SectorProduct(U1(0), SU2(1)) != SectorProduct(U1(1), SU2(1))
    @test SectorProduct(U1(1)) != U1(1)
    @test SectorProduct(U1(1)) == SectorProduct(U1(1), U1(0))
    @test SectorProduct(U1(1)) != SectorProduct(U1(1), U1(1))
    @test SectorProduct(U1(0), SU2(0)) == TrivialSector()
    @test SectorProduct(U1(0), SU2(0)) == SectorProduct(TrivialSector(), SU2(0))
    @test SectorProduct(U1(0), SU2(0)) == SectorProduct(U1(0), TrivialSector())
    @test SectorProduct(U1(0), SU2(0)) == SectorProduct(TrivialSector(), TrivialSector())

    @test SectorProduct(U1(0)) < SectorProduct((U1(1)))
    @test SectorProduct(U1(0), U1(2)) < SectorProduct((U1(1)), U1(0))
    @test SectorProduct(U1(0)) < SectorProduct(U1(0), U1(1))
    @test SectorProduct(U1(0)) > SectorProduct(U1(0), U1(-1))
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
    p1 = SectorProduct(U1(1))
    p2 = SectorProduct(U1(2))
    @test (@inferred p1 ⊗ TrivialSector()) == p1
    @test (@inferred TrivialSector() ⊗ p2) == p2
    @test (@inferred p1 ⊗ p2) == SectorProduct(U1(3))

    p11 = U1(1) × U1(1)
    @test p11 ⊗ p11 == U1(2) × U1(2)

    p123 = U1(1) × U1(2) × U1(3)
    @test p123 ⊗ p123 == U1(2) × U1(4) × U1(6)

    s1 = SectorProduct(U1(1), Z{2}(1))
    s2 = SectorProduct(U1(0), Z{2}(0))
    @test s1 ⊗ s2 == U1(1) × Z{2}(1)
  end

  @testset "Fusion of NonAbelian products" begin
    p0 = SectorProduct(SU2(0))
    ph = SectorProduct(SU2(1//2))
    @test space_isequal(
      (@inferred p0 ⊗ TrivialSector()), gradedrange([SectorProduct(SU2(0)) => 1])
    )
    @test space_isequal(
      (@inferred TrivialSector() ⊗ ph), gradedrange([SectorProduct(SU2(1//2)) => 1])
    )

    phh = SU2(1//2) × SU2(1//2)
    @test space_isequal(
      phh ⊗ phh,
      gradedrange([
        (SU2(0) × SU2(0)) => 1,
        (SU2(1) × SU2(0)) => 1,
        (SU2(0) × SU2(1)) => 1,
        (SU2(1) × SU2(1)) => 1,
      ]),
    )
    @test space_isequal(
      phh ⊗ phh,
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
    @test space_isequal(s ⊗ s, gradedrange([s => 1]))

    s = τ × τ
    @test space_isequal(
      s ⊗ s, gradedrange([(ı × ı) => 1, (τ × ı) => 1, (ı × τ) => 1, (τ × τ) => 1])
    )

    σ = Ising("σ")
    ψ = Ising("ψ")
    s = τ × σ
    g = gradedrange([
      (ı × Ising("1")) => 1, (τ × Ising("1")) => 1, (ı × ψ) => 1, (τ × ψ) => 1
    ])
    @test space_isequal(s ⊗ s, g)
  end

  @testset "Fusion of mixed Abelian and NonAbelian products" begin
    p2h = U1(2) × SU2(1//2)
    p1h = U1(1) × SU2(1//2)
    @test space_isequal(
      p2h ⊗ p1h, gradedrange([(U1(3) × SU2(0)) => 1, (U1(3) × SU2(1)) => 1])
    )

    p1h1 = U1(1) × SU2(1//2) × Z{2}(1)
    @test space_isequal(
      p1h1 ⊗ p1h1,
      gradedrange([(U1(2) × SU2(0) × Z{2}(0)) => 1, (U1(2) × SU2(1) × Z{2}(0)) => 1]),
    )
  end

  @testset "Fusion of fully mixed products" begin
    s = U1(1) × SU2(1//2) × Ising("σ")
    @test space_isequal(
      s ⊗ s,
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
    @test space_isequal(
      s ⊗ s,
      gradedrange([
        (SU2(0) × U1(2) × ı) => 1,
        (SU2(1) × U1(2) × ı) => 1,
        (SU2(0) × U1(2) × τ) => 1,
        (SU2(1) × U1(2) × τ) => 1,
      ]),
    )

    s = U1(1) × ı × τ
    @test space_isequal(s ⊗ s, gradedrange([(U1(2) × ı × ı) => 1, (U1(2) × ı × τ) => 1]))
  end

  @testset "Fusion of different length Categories" begin
    @test SectorProduct(U1(1) × U1(0)) ⊗ SectorProduct(U1(1)) ==
      SectorProduct(U1(2) × U1(0))
    @test space_isequal(
      (@inferred SectorProduct(SU2(0) × SU2(0)) ⊗ SectorProduct(SU2(1))),
      gradedrange([SectorProduct(SU2(1) × SU2(0)) => 1]),
    )

    @test space_isequal(
      (@inferred SectorProduct(SU2(1) × U1(1)) ⊗ SectorProduct(SU2(0))),
      gradedrange([SectorProduct(SU2(1) × U1(1)) => 1]),
    )
    @test space_isequal(
      (@inferred SectorProduct(U1(1) × SU2(1)) ⊗ SectorProduct(U1(2))),
      gradedrange([SectorProduct(U1(3) × SU2(1)) => 1]),
    )

    # check incompatible sectors
    p12 = Z{2}(1) × U1(2)
    z12 = Z{2}(1) × Z{2}(1)
    @test_throws MethodError p12 ⊗ z12
  end

  @testset "GradedUnitRange fusion rules" begin
    s1 = U1(1) × SU2(1//2) × Ising("σ")
    s2 = U1(0) × SU2(1//2) × Ising("1")
    g1 = gradedrange([s1 => 2])
    g2 = gradedrange([s2 => 1])
    @test space_isequal(
      fusion_product(g1, g2),
      gradedrange([U1(1) × SU2(0) × Ising("σ") => 2, U1(1) × SU2(1) × Ising("σ") => 2]),
    )
  end
end

@testset "Test Named Sector Products" begin
  @testset "Construct from × of NamedTuples" begin
    s = (A=U1(1),) × (B=Z{2}(0),)
    @test length(arguments(s)) == 2
    @test arguments(s)[:A] == U1(1)
    @test arguments(s)[:B] == Z{2}(0)
    @test (@inferred quantum_dimension(s)) == 1
    @test (@inferred dual(s)) == (A=U1(-1),) × (B=Z{2}(0),)
    @test (@inferred trivial(s)) == (A=U1(0),) × (B=Z{2}(0),)

    s = (A=U1(1),) × (B=SU2(2),)
    @test length(arguments(s)) == 2
    @test arguments(s)[:A] == U1(1)
    @test arguments(s)[:B] == SU2(2)
    @test (@inferred quantum_dimension(s)) == 5
    @test (@inferred dual(s)) == (A=U1(-1),) × (B=SU2(2),)
    @test (@inferred trivial(s)) == (A=U1(0),) × (B=SU2(0),)
    @test s == (B=SU2(2),) × (A=U1(1),)

    s = s × (C=Ising("ψ"),)
    @test length(arguments(s)) == 3
    @test arguments(s)[:C] == Ising("ψ")
    @test (@inferred quantum_dimension(s)) == 5.0
    @test (@inferred dual(s)) == (A=U1(-1),) × (B=SU2(2),) × (C=Ising("ψ"),)

    s1 = (A=U1(1),) × (B=Z{2}(0),)
    s2 = (A=U1(1),) × (C=Z{2}(0),)
    @test_throws ArgumentError s1 × s2
  end

  @testset "Construct from Pairs" begin
    s = SectorProduct("A" => U1(2))
    @test length(arguments(s)) == 1
    @test arguments(s)[:A] == U1(2)
    @test s == SectorProduct(; A=U1(2))
    @test (@inferred quantum_dimension(s)) == 1
    @test (@inferred dual(s)) == SectorProduct("A" => U1(-2))
    @test (@inferred trivial(s)) == SectorProduct(; A=U1(0))

    s = SectorProduct("B" => Ising("ψ"), :C => Z{2}(1))
    @test length(arguments(s)) == 2
    @test arguments(s)[:B] == Ising("ψ")
    @test arguments(s)[:C] == Z{2}(1)
    @test (@inferred quantum_dimension(s)) == 1.0
  end

  @testset "Comparisons with unspecified labels" begin
    # convention: arguments evaluate as equal if unmatched labels are trivial
    # this is different from ordered tuple convention
    q2 = SectorProduct(; N=U1(2))
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
    g = gradedrange([
      SectorProduct(; A=U1(0), B=Z{2}(0)) => 1, SectorProduct(; A=U1(1), B=Z{2}(0)) => 2
    ])  # abelian
    @test (@inferred quantum_dimension(g)) == 3

    g = gradedrange([  # non-abelian
      SectorProduct(; A=SU2(0), B=SU2(0)) => 1,
      SectorProduct(; A=SU2(1), B=SU2(0)) => 1,
      SectorProduct(; A=SU2(0), B=SU2(1)) => 1,
      SectorProduct(; A=SU2(1), B=SU2(1)) => 1,
    ])
    @test (@inferred quantum_dimension(g)) == 16

    # mixed group
    g = gradedrange([
      SectorProduct(; A=U1(2), B=SU2(0), C=Z{2}(0)) => 1,
      SectorProduct(; A=U1(2), B=SU2(1), C=Z{2}(0)) => 1,
    ])
    @test (@inferred quantum_dimension(g)) == 4
    g = gradedrange([
      SectorProduct(; A=SU2(0), B=Z{2}(0), C=SU2(1//2)) => 1,
      SectorProduct(; A=SU2(0), B=Z{2}(1), C=SU2(1//2)) => 1,
    ])
    @test (@inferred quantum_dimension(g)) == 4

    # non group sectors
    g_fib = gradedrange([SectorProduct(; A=Fib("1"), B=Fib("1")) => 1])
    g_ising = gradedrange([SectorProduct(; A=Ising("1"), B=Ising("1")) => 1])
    @test (@inferred quantum_dimension(g_fib)) == 1.0
    @test (@inferred quantum_dimension(g_ising)) == 1.0

    # mixed product Abelian / NonAbelian / NonGroup
    g = gradedrange([
      SectorProduct(; A=U1(2), B=SU2(0), C=Ising(1)) => 1,
      SectorProduct(; A=U1(2), B=SU2(1), C=Ising(1)) => 1,
      SectorProduct(; A=U1(2), B=SU2(0), C=Ising("ψ")) => 1,
      SectorProduct(; A=U1(2), B=SU2(1), C=Ising("ψ")) => 1,
    ])
    @test (@inferred quantum_dimension(g)) == 8.0

    g = gradedrange([
      SectorProduct(; A=Fib("1"), B=SU2(0), C=U1(2)) => 1,
      SectorProduct(; A=Fib("1"), B=SU2(1), C=U1(2)) => 1,
      SectorProduct(; A=Fib("τ"), B=SU2(0), C=U1(2)) => 1,
      SectorProduct(; A=Fib("τ"), B=SU2(1), C=U1(2)) => 1,
    ])
    @test (@inferred quantum_dimension(g)) == 4.0 + 4.0quantum_dimension(Fib("τ"))
  end

  @testset "Fusion of Abelian products" begin
    q00 = SectorProduct(;)
    q10 = SectorProduct(; A=U1(1))
    q01 = SectorProduct(; B=U1(1))
    q11 = SectorProduct(; A=U1(1), B=U1(1))

    @test (@inferred q10 ⊗ q10) == SectorProduct(; A=U1(2))
    @test (@inferred q01 ⊗ q00) == q01
    @test (@inferred q00 ⊗ q01) == q01
    @test (@inferred q10 ⊗ q01) == q11
    @test q11 ⊗ q11 == SectorProduct(; A=U1(2), B=U1(2))

    s11 = SectorProduct(; A=U1(1), B=Z{2}(1))
    s10 = SectorProduct(; A=U1(1))
    s01 = SectorProduct(; B=Z{2}(1))
    @test (@inferred s01 ⊗ q00) == s01
    @test (@inferred q00 ⊗ s01) == s01
    @test (@inferred s10 ⊗ s01) == s11
    @test s11 ⊗ s11 == SectorProduct(; A=U1(2), B=Z{2}(0))
  end

  @testset "Fusion of NonAbelian products" begin
    p0 = SectorProduct(;)
    pha = SectorProduct(; A=SU2(1//2))
    phb = SectorProduct(; B=SU2(1//2))
    phab = SectorProduct(; A=SU2(1//2), B=SU2(1//2))

    @test space_isequal(
      (@inferred pha ⊗ pha),
      gradedrange([SectorProduct(; A=SU2(0)) => 1, SectorProduct(; A=SU2(1)) => 1]),
    )
    @test space_isequal((@inferred pha ⊗ p0), gradedrange([pha => 1]))
    @test space_isequal((@inferred p0 ⊗ phb), gradedrange([phb => 1]))
    @test space_isequal((@inferred pha ⊗ phb), gradedrange([phab => 1]))

    @test space_isequal(
      phab ⊗ phab,
      gradedrange([
        SectorProduct(; A=SU2(0), B=SU2(0)) => 1,
        SectorProduct(; A=SU2(1), B=SU2(0)) => 1,
        SectorProduct(; A=SU2(0), B=SU2(1)) => 1,
        SectorProduct(; A=SU2(1), B=SU2(1)) => 1,
      ]),
    )
  end

  @testset "Fusion of NonGroupCategory products" begin
    ı = Fib("1")
    τ = Fib("τ")
    s = SectorProduct(; A=ı, B=ı)
    @test space_isequal(s ⊗ s, gradedrange([s => 1]))

    s = SectorProduct(; A=τ, B=τ)
    @test space_isequal(
      s ⊗ s,
      gradedrange([
        SectorProduct(; A=ı, B=ı) => 1,
        SectorProduct(; A=τ, B=ı) => 1,
        SectorProduct(; A=ı, B=τ) => 1,
        SectorProduct(; A=τ, B=τ) => 1,
      ]),
    )

    σ = Ising("σ")
    ψ = Ising("ψ")
    s = SectorProduct(; A=τ, B=σ)
    g = gradedrange([
      SectorProduct(; A=ı, B=Ising("1")) => 1,
      SectorProduct(; A=τ, B=Ising("1")) => 1,
      SectorProduct(; A=ı, B=ψ) => 1,
      SectorProduct(; A=τ, B=ψ) => 1,
    ])
    @test space_isequal(s ⊗ s, g)
  end

  @testset "Fusion of mixed Abelian and NonAbelian products" begin
    q0h = SectorProduct(; J=SU2(1//2))
    q10 = (N=U1(1),) × (J=SU2(0),)
    # Put names in reverse order sometimes:
    q1h = (J=SU2(1//2),) × (N=U1(1),)
    q11 = (N=U1(1),) × (J=SU2(1),)
    q20 = (N=U1(2),) × (J=SU2(0),)  # julia 1.6 does not accept gradedrange without J
    q2h = (N=U1(2),) × (J=SU2(1//2),)
    q21 = (N=U1(2),) × (J=SU2(1),)
    q22 = (N=U1(2),) × (J=SU2(2),)

    @test space_isequal(q1h ⊗ q1h, gradedrange([q20 => 1, q21 => 1]))
    @test space_isequal(q10 ⊗ q1h, gradedrange([q2h => 1]))
    @test space_isequal((@inferred q0h ⊗ q1h), gradedrange([q10 => 1, q11 => 1]))
    @test space_isequal(q11 ⊗ q11, gradedrange([q20 => 1, q21 => 1, q22 => 1]))
  end

  @testset "Fusion of fully mixed products" begin
    s = SectorProduct(; A=U1(1), B=SU2(1//2), C=Ising("σ"))
    @test space_isequal(
      s ⊗ s,
      gradedrange([
        SectorProduct(; A=U1(2), B=SU2(0), C=Ising("1")) => 1,
        SectorProduct(; A=U1(2), B=SU2(1), C=Ising("1")) => 1,
        SectorProduct(; A=U1(2), B=SU2(0), C=Ising("ψ")) => 1,
        SectorProduct(; A=U1(2), B=SU2(1), C=Ising("ψ")) => 1,
      ]),
    )

    ı = Fib("1")
    τ = Fib("τ")
    s = SectorProduct(; A=SU2(1//2), B=U1(1), C=τ)
    @test space_isequal(
      s ⊗ s,
      gradedrange([
        SectorProduct(; A=SU2(0), B=U1(2), C=ı) => 1,
        SectorProduct(; A=SU2(1), B=U1(2), C=ı) => 1,
        SectorProduct(; A=SU2(0), B=U1(2), C=τ) => 1,
        SectorProduct(; A=SU2(1), B=U1(2), C=τ) => 1,
      ]),
    )

    s = SectorProduct(; A=τ, B=U1(1), C=ı)
    @test space_isequal(
      s ⊗ s,
      gradedrange([
        SectorProduct(; B=U1(2), A=ı, C=ı) => 1, SectorProduct(; B=U1(2), A=τ, C=ı) => 1
      ]),
    )
  end
  @testset "GradedUnitRange fusion rules" begin
    s1 = SectorProduct(; A=U1(1), B=SU2(1//2), C=Ising("σ"))
    s2 = SectorProduct(; A=U1(0), B=SU2(1//2), C=Ising("1"))
    g1 = gradedrange([s1 => 2])
    g2 = gradedrange([s2 => 1])
    s3 = SectorProduct(; A=U1(1), B=SU2(0), C=Ising("σ"))
    s4 = SectorProduct(; A=U1(1), B=SU2(1), C=Ising("σ"))
    @test space_isequal(fusion_product(g1, g2), gradedrange([s3 => 2, s4 => 2]))

    sA = SectorProduct(; A=U1(1))
    sB = SectorProduct(; B=SU2(1//2))
    sAB = SectorProduct(; A=U1(1), B=SU2(1//2))
    gA = gradedrange([sA => 2])
    gB = gradedrange([sB => 1])
    @test space_isequal(fusion_product(gA, gB), gradedrange([sAB => 2]))
  end
end

@testset "Mixing implementations" begin
  st1 = SectorProduct(U1(1))
  sA1 = SectorProduct(; A=U1(1))

  @test sA1 != st1
  @test_throws MethodError sA1 < st1
  @test_throws MethodError st1 < sA1
  @test_throws MethodError st1 ⊗ sA1
  @test_throws MethodError sA1 ⊗ st1
  @test_throws ArgumentError st1 × sA1
  @test_throws ArgumentError sA1 × st1
end

@testset "Empty SymmetrySector" begin
  st1 = SectorProduct(U1(1))
  sA1 = SectorProduct(; A=U1(1))

  for s in (SectorProduct(()), SectorProduct((;)))
    @test s == TrivialSector()
    @test s == SectorProduct(())
    @test s == SectorProduct((;))

    @test !(s < SectorProduct())
    @test !(s < SectorProduct(;))

    @test (@inferred s × SectorProduct(())) == s
    @test (@inferred s × SectorProduct((;))) == s
    @test (@inferred s ⊗ SectorProduct(())) == s
    @test (@inferred s ⊗ SectorProduct((;))) == s

    @test (@inferred dual(s)) == s
    @test (@inferred trivial(s)) == s
    @test (@inferred quantum_dimension(s)) == 1

    g0 = gradedrange([s => 2])
    @test space_isequal((@inferred fusion_product(g0, g0)), gradedrange([s => 4]))

    @test (@inferred s × U1(1)) == st1
    @test (@inferred U1(1) × s) == st1
    @test (@inferred s × st1) == st1
    @test (@inferred st1 × s) == st1
    @test (@inferred s × sA1) == sA1
    @test (@inferred sA1 × s) == sA1

    @test (@inferred U1(1) ⊗ s) == st1
    @test (@inferred s ⊗ U1(1)) == st1
    @test (@inferred SU2(0) ⊗ s) == gradedrange([SectorProduct(SU2(0)) => 1])
    @test (@inferred s ⊗ SU2(0)) == gradedrange([SectorProduct(SU2(0)) => 1])
    @test (@inferred Fib("τ") ⊗ s) == gradedrange([SectorProduct(Fib("τ")) => 1])
    @test (@inferred s ⊗ Fib("τ")) == gradedrange([SectorProduct(Fib("τ")) => 1])

    @test (@inferred st1 ⊗ s) == st1
    @test (@inferred SectorProduct(SU2(0)) ⊗ s) == gradedrange([SectorProduct(SU2(0)) => 1])
    @test (@inferred SectorProduct(Fib("τ"), SU2(1), U1(2)) ⊗ s) ==
      gradedrange([SectorProduct(Fib("τ"), SU2(1), U1(2)) => 1])

    @test (@inferred sA1 ⊗ s) == sA1
    @test (@inferred SectorProduct(; A=SU2(0)) ⊗ s) ==
      gradedrange([SectorProduct(; A=SU2(0)) => 1])
    @test (@inferred SectorProduct(; A=Fib("τ"), B=SU2(1), C=U1(2)) ⊗ s) ==
      gradedrange([SectorProduct(; A=Fib("τ"), B=SU2(1), C=U1(2)) => 1])

    # Empty behaves as empty NamedTuple
    @test s != U1(0)
    @test s == SectorProduct(U1(0))
    @test s == SectorProduct(; A=U1(0))
    @test SectorProduct(; A=U1(0)) == s
    @test s != sA1
    @test s != st1

    @test s < st1
    @test SectorProduct(U1(-1)) < s
    @test s < sA1
    @test s > SectorProduct(; A=U1(-1))
    @test !(s < SectorProduct(; A=U1(0)))
    @test !(s > SectorProduct(; A=U1(0)))
  end
end
end
