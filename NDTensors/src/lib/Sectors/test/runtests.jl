import NDTensors.Sectors:
  ⊗,
  ⊕,
  category,
  Fib,
  Ising,
  istrivial,
  Label,
  nactive,
  name,
  Sector,
  SU,
  SUd,
  SUz,
  U,
  values,
  Z
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

  @testset "Ƶ_2" begin
    z0 = Label(Z(2), 0)
    z1 = Label(Z(2), 1)

    @test z0 ⊗ z0 == [z0]
    @test z0 ⊗ z1 == [z1]
    @test z1 ⊗ z0 == [z1]
    @test z1 ⊗ z1 == [z0]
  end

  @testset "Ƶ_3" begin
    z0 = Label(Z(3), 0)
    z1 = Label(Z(3), 1)
    z2 = Label(Z(3), 2)

    @test z0 ⊗ z0 == [z0]
    @test z0 ⊗ z1 == [z1]
    @test z0 ⊗ z2 == [z2]
    @test z1 ⊗ z0 == [z1]
    @test z1 ⊗ z1 == [z2]
    @test z1 ⊗ z2 == [z0]
  end

  @testset "SU(2)" begin
    j0 = Label(SU(2), 0)
    j½ = Label(SU(2), 1//2)
    j1 = Label(SU(2), 1)
    j3_2 = Label(SU(2), 3//2)
    j2 = Label(SU(2), 2)

    @test istrivial(j0)
    @test !istrivial(j½)
    @test !istrivial(j1)

    @test j0 ⊗ j0 == [j0]
    @test j0 ⊗ j½ == [j½]
    @test j0 ⊗ j1 == [j1]
    @test j½ ⊗ j½ == [j0, j1]
    @test j½ ⊗ j½ == j0 ⊕ j1
    @test j½ ⊗ j1 == j½ ⊕ j3_2
  end

  #
  # SUd(2) is the group SU(2)
  # but with values corresponding to
  # the dimension (d=2j+1) of 
  # each representation
  #
  @testset "SUd(2)" begin
    d1 = Label(SUd(2), 1) # spin 0
    d2 = Label(SUd(2), 2) # spin 1/2
    d3 = Label(SUd(2), 3) # spin 1
    d4 = Label(SUd(2), 4) # spin 3/2
    d5 = Label(SUd(2), 5) # spin 2

    @test d1 ⊗ d1 == [d1]
    @test d1 ⊗ d2 == [d2]
    @test d1 ⊗ d3 == [d3]

    @test d2 ⊗ d2 == d1 ⊕ d3
    @test d2 ⊗ d3 == d2 ⊕ d4
    @test d3 ⊗ d3 == d1 ⊕ d3 ⊕ d5
  end

  @testset "Ising" begin
    ı = Label(Ising, "1")
    σ = Label(Ising, "σ")
    ψ = Label(Ising, "ψ")

    @test ı ⊗ ı == [ı]
    @test ı ⊗ σ == [σ]
    @test ı ⊗ ψ == [ψ]
    @test σ ⊗ σ == ı ⊕ ψ
    @test ψ ⊗ σ == [σ]
    @test ψ ⊗ ψ == [ı]
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
    q½p = Label(SUz(2), (1//2, 1//2))
    q½m = Label(SUz(2), (1//2, -1//2))
    q1p = Label(SUz(2), (1, +1))
    q10 = Label(SUz(2), (1, 0))
    q1m = Label(SUz(2), (1, -1))
    q00 = Label(SUz(2), (0, 0))
    q22 = Label(SUz(2), (2, +2))
    q21 = Label(SUz(2), (2, +1))
    q20 = Label(SUz(2), (2, 0))

    @test q½p ⊗ q½p == [q1p]
    @test q½p ⊗ q½m == q00 ⊕ q10
    @test q½m ⊗ q½m == [q1m]

    @test q1p ⊗ q1p == [q22]
    @test q1p ⊗ q10 == q1p ⊕ q21
    @test q1p ⊗ q1m == q00 ⊕ q10 ⊕ q20
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
    # Put names in reverse order sometimes:
    q1h = Sector(("J", SU(2), 1//2), ("N", U(1), 1))
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
