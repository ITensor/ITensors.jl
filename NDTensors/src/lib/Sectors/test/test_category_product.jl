import NDTensors.Sectors: ⊗, ⊕, ×, Fib, labels, Ising, Sector, SU, SU2, U1, Z
using Test

@testset "Test Named Sectors" begin
  @testset "Construct from × of NamedTuples" begin
    s = (A=U1(1),) × (B=SU2(2),)
    @test length(s) == 2
    @test s[:A] == U1(1)
    @test s[:B] == SU2(2)

    s = s × (C=Ising("ψ"),)
    @test length(s) == 3
    @test s[:C] == Ising("ψ")
  end

  @testset "Construct from Pairs" begin
    s = Sector("A" => U1(2))
    @test length(s) == 1
    @test s[:A] == U1(2)
    @test s == Sector(; A=U1(2))

    s = Sector("B" => Ising("ψ"), :C => Z{2}(1))
    @test length(s) == 2
    @test s[:B] == Ising("ψ")
    @test s[:C] == Z{2}(1)
  end

  @testset "Multiple U(1)'s" begin
    q00 = Sector()
    q10 = Sector(; A=U1(1))
    q01 = Sector(; B=U1(1))
    q11 = Sector(; A=U1(1), B=U1(1))

    @test q00 ⊗ q00 == [q00]
    @test q01 ⊗ q00 == [q01]
    @test q00 ⊗ q01 == [q01]
    @test q10 ⊗ q01 == [q11]
  end

  @testset "U(1) ⊗ SU(2) conventional" begin
    q0 = Sector()
    q0h = Sector(; J=SU2(1//2))
    q10 = (N=U1(1),) × (J=SU2(0),)
    # Put names in reverse order sometimes:
    q1h = (J=SU2(1//2),) × (N=U1(1),)
    q11 = (N=U1(1),) × (J=SU2(1),)
    q20 = Sector(; N=U1(2))
    q2h = (N=U1(2),) × (J=SU2(1//2),)
    q21 = (N=U1(2),) × (J=SU2(1),)
    q22 = (N=U1(2),) × (J=SU2(2),)

    @test q1h ⊗ q1h == q20 ⊕ q21
    @test q10 ⊗ q1h == [q2h]
    @test q0h ⊗ q1h == q10 ⊕ q11
    @test q11 ⊗ q11 == q20 ⊕ q21 ⊕ q22
  end

  @testset "U(1) ⊗ SU(2)" begin
    q0 = Sector()
    q0h = Sector(; J=SU{2}(2))
    q10 = (N=U1(1),) × (J=SU{2}(1),)
    # Put names in reverse order sometimes:
    q1h = (J=SU{2}(2),) × (N=U1(1),)
    q11 = (N=U1(1),) × (J=SU{2}(3),)
    q20 = Sector(; N=U1(2))
    q2h = (N=U1(2),) × (J=SU{2}(2),)
    q21 = (N=U1(2),) × (J=SU{2}(3),)
    q22 = (N=U1(2),) × (J=SU{2}(5),)

    @test q1h ⊗ q1h == q20 ⊕ q21
    @test q10 ⊗ q1h == [q2h]
    @test q0h ⊗ q1h == q10 ⊕ q11
    @test q11 ⊗ q11 == q20 ⊕ q21 ⊕ q22
  end

  @testset "Comparisons with unspecified labels" begin
    q2 = Sector(; N=U1(2))
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

@testset "Test Ordered Sectors" begin
  @testset "Ordered Constructor" begin
    s = Sector(U1(1), U1(2))
    @test length(s) == 2
    @test s[1] == U1(1)
    @test s[2] == U1(2)

    s = U1(1) × SU2(1//2) × U1(3)
    @test length(s) == 3
    @test s[1] == U1(1)
    @test s[2] == SU2(1//2)
    @test s[3] == U1(3)
  end

  @testset "Fusion of U1 products" begin
    p11 = U1(1) × U1(1)
    @test p11 ⊗ p11 == [U1(2) × U1(2)]

    p123 = U1(1) × U1(2) × U1(3)
    @test p123 ⊗ p123 == [U1(2) × U1(4) × U1(6)]
  end

  @testset "Enforce same number of spaces" begin
    p12 = U1(1) × U1(2)
    p123 = U1(1) × U1(2) × U1(3)
    @test_throws DimensionMismatch p12 ⊗ p123
  end

  @testset "Fusion of SU2 products" begin
    phh = SU2(1//2) × SU2(1//2)
    @test phh ⊗ phh ==
      (SU2(0) × SU2(0)) ⊕ (SU2(1) × SU2(0)) ⊕ (SU2(0) × SU2(1)) ⊕ (SU2(1) × SU2(1))
  end

  @testset "Fusion of mixed U1 and SU2 products" begin
    p2h = U1(2) × SU2(1//2)
    p1h = U1(1) × SU2(1//2)
    @test p2h ⊗ p1h == (U1(3) × SU2(0)) ⊕ (U1(3) × SU2(1))

    p1h1 = U1(1) × SU2(1//2) × Z{2}(1)
    @test p1h1 ⊗ p1h1 == (U1(2) × SU2(0) × Z{2}(0)) ⊕ (U1(2) × SU2(1) × Z{2}(0))
  end
end

nothing
