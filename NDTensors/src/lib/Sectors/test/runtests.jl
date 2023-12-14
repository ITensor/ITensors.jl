import NDTensors.Sectors:
  ⊗,
  ⊕,
  U,
  SU,
  SUd,
  SUz,
  Z,
  Fib,
  Ising,
  Sector,
  nactive,
  Label,
  category,
  name,
  istrivial,
  values
using Test

@testset "Test Label system" begin
  @testset "U(1) Labels" begin
    l0 = Label(U(1), 0)
    l1 = Label(U(1), 1)
    l2 = Label(U(1), 2)

    @test name(l1) == ""
    @test category(l1) == U(1)
    @test values(l1) == (1, 0)

    @test istrivial(l0)
    @test !istrivial(l1)
    @test !istrivial(l2)

    @test l0 ⊗ l0 == [l0]
    @test l0 ⊗ l1 == [l1]
    @test l0 ⊗ l2 == [l2]
    @test l1 ⊗ l1 == [l2]
  end

  @testset "SU(2) Labels" begin
    l0 = Label(SU(2), 0)
    l½ = Label(SU(2), 1//2)
    l1 = Label(SU(2), 1)
    l3_2 = Label(SU(2), 3//2)

    @test istrivial(l0)
    @test !istrivial(l½)
    @test !istrivial(l1)

    @test l0 ⊗ l½ == [l½]
    @test l0 ⊗ l1 == [l1]
    @test l½ ⊗ l½ == l0 ⊕ l1
    @test l½ ⊗ l1 == l½ ⊕ l3_2
  end

  @testset "Named Labels" begin
    n0 = Label("N", U(1), 0)
    n1 = Label("N", U(1), 1)
    @test n0 ⊗ n0 == [n0]
    @test n0 ⊗ n1 == [n1]
    @test name(first(n0 ⊗ n0)) == "N"
  end
end

@testset "Test Sector System" begin
  @testset "U(1)" begin
    q0 = Sector(U(1), 0)
    q1 = Sector(U(1), 1)
    q2 = Sector(U(1), 2)
    q3 = Sector(U(1), 3)

    @test q0 ⊗ q0 == [q0]
    @test q0 ⊗ q1 == [q1]
    @test q1 ⊗ q0 == [q1]
    @test q0 ⊗ q2 == [q2]
    @test q1 ⊗ q2 == [q3]
  end

  @testset "Ƶ_2" begin
    z0 = Sector(Z(2), 0)
    z1 = Sector(Z(2), 1)

    @test z0 ⊗ z0 == [z0]
    @test z0 ⊗ z1 == [z1]
    @test z1 ⊗ z0 == [z1]
    @test z1 ⊗ z1 == [z0]
  end

  @testset "Ƶ_3" begin
    z0 = Sector(Z(3), 0)
    z1 = Sector(Z(3), 1)
    z2 = Sector(Z(3), 2)

    @test z0 ⊗ z0 == [z0]
    @test z0 ⊗ z1 == [z1]
    @test z0 ⊗ z2 == [z2]
    @test z1 ⊗ z0 == [z1]
    @test z1 ⊗ z1 == [z2]
    @test z1 ⊗ z2 == [z0]
  end

  @testset "SU(2)" begin
    j0 = Sector(SU(2), 0)
    j½ = Sector(SU(2), 1//2)
    j1 = Sector(SU(2), 1)
    j3_2 = Sector(SU(2), 3//2)
    j2 = Sector(SU(2), 2)

    @test j0 ⊗ j0 == [j0]
    @test j0 ⊗ j½ == [j½]
    @test j0 ⊗ j1 == [j1]

    @test j½ ⊗ j½ == [j0, j1]
    @test j½ ⊗ j½ == j0 ⊕ j1
    @test j½ ⊗ j1 == j½ ⊕ j3_2
    @test j1 ⊗ j1 == j0 ⊕ j1 ⊕ j2
  end

  #
  # SUd(2) is the group SU(2)
  # but with values corresponding to
  # the dimension (d=2j+1) of 
  # each representation
  #
  @testset "SUd(2)" begin
    d1 = Sector(SUd(2), 1) # spin 0
    d2 = Sector(SUd(2), 2) # spin 1/2
    d3 = Sector(SUd(2), 3) # spin 1
    d4 = Sector(SUd(2), 4) # spin 3/2
    d5 = Sector(SUd(2), 5) # spin 2

    @test d1 ⊗ d1 == [d1]
    @test d1 ⊗ d2 == [d2]
    @test d1 ⊗ d3 == [d3]

    @test d2 ⊗ d2 == d1 ⊕ d3
    @test d2 ⊗ d3 == d2 ⊕ d4
    @test d3 ⊗ d3 == d1 ⊕ d3 ⊕ d5
  end

  @testset "Ising" begin
    ı = Sector(Ising, "1")
    σ = Sector(Ising, "σ")
    ψ = Sector(Ising, "ψ")

    @test ı ⊗ ı == [ı]
    @test ı ⊗ σ == [σ]
    @test ı ⊗ ψ == [ψ]
    @test σ ⊗ σ == ı ⊕ ψ
    @test ψ ⊗ σ == [σ]
    @test ψ ⊗ ψ == [ı]

    @test Sector(Ising, 0) == ı
    @test Sector(Ising, 1//2) == σ
    @test Sector(Ising, 1) == ψ
  end

  @testset "Fibonacci" begin
    ı = Sector(Fib, "1")
    τ = Sector(Fib, "τ")

    @test ı ⊗ ı == [ı]
    @test ı ⊗ τ == [τ]
    @test τ ⊗ τ == ı ⊕ τ

    @test Sector(Fib, 0) == ı
    @test Sector(Fib, 1) == τ
  end

  @testset "SU(2) with z component" begin
    q½p = Sector("J", SUz(2), (1//2, 1//2))
    q½m = Sector("J", SUz(2), (1//2, -1//2))
    q1p = Sector("J", SUz(2), (1, +1))
    q10 = Sector("J", SUz(2), (1, 0))
    q1m = Sector("J", SUz(2), (1, -1))
    q00 = Sector("J", SUz(2), (0, 0))
    q22 = Sector("J", SUz(2), (2, +2))
    q21 = Sector("J", SUz(2), (2, +1))
    q20 = Sector("J", SUz(2), (2, 0))

    @test q½p ⊗ q½p == [q1p]
    @test q½p ⊗ q½m == q00 ⊕ q10
    @test q½m ⊗ q½m == [q1m]

    @test q1p ⊗ q1p == [q22]
    @test q1p ⊗ q10 == q1p ⊕ q21
    @test q1p ⊗ q1m == q00 ⊕ q10 ⊕ q20
  end

  @testset "Multiple U(1)'s" begin
    q00 = Sector()
    q10 = Sector("A", U(1), 1)
    q01 = Sector("B", U(1), 1)
    q11 = Sector(("A", U(1), 1), ("B", U(1), 1))

    @test q00 ⊗ q00 == [q00]
    @test q01 ⊗ q00 == [q01]
    @test q00 ⊗ q01 == [q01]
    @test q10 ⊗ q01 == [q11]
  end

  @testset "U(1) ⊗ SU(2)" begin
    q0 = Sector()
    q0h = Sector(("J", SU(2), 1//2))
    q10 = Sector(("N", U(1), 1), ("J", SU(2), 0))
    q1h = Sector(("N", U(1), 1), ("J", SU(2), 1//2))
    q11 = Sector(("N", U(1), 1), ("J", SU(2), 1))
    q20 = Sector(("N", U(1), 2))
    q2h = Sector(("N", U(1), 2), ("J", SU(2), 1//2))
    q21 = Sector(("N", U(1), 2), ("J", SU(2), 1))
    q22 = Sector(("N", U(1), 2), ("J", SU(2), 2))

    @test q1h ⊗ q1h == q20 ⊕ q21
    @test q10 ⊗ q1h == [q2h]
    @test q0h ⊗ q1h == q10 ⊕ q11
    @test q11 ⊗ q11 == q20 ⊕ q21 ⊕ q22
  end

  @testset "Set all sectors" begin
    q = Sector(("A", U(1), 1))
    @test nactive(q) == 1
    q = Sector(("B", U(1), 2), ("A", U(1), 1))
    @test nactive(q) == 2
    q = Sector(("C", U(1), 3), ("B", U(1), 2), ("A", U(1), 1))
    @test nactive(q) == 3
    q = Sector(("D", U(1), 4), ("C", U(1), 3), ("B", U(1), 2), ("A", U(1), 1))
    @test nactive(q) == 4
  end

  @testset "Comparison with unspecified labels" begin
    q2 = Sector("N", U(1), 2)

    q20 = Sector(("N", U(1), 2), ("J", SU(2), 0))

    @test q20 == q2

    q21 = Sector(("N", U(1), 2), ("J", SU(2), 1))
    @test q21 != q2

    a = Sector(("A", U(1), 0), ("B", U(1), 2))
    b = Sector(("B", U(1), 2), ("C", U(1), 0))
    @test a == b
    c = Sector(("B", U(1), 2), ("C", U(1), 1))
    @test a != c
  end
end

nothing
