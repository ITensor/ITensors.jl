import NDTensors.Sectors: ⊗, ⊕, U, SU, SUd, SUz, Z, Fib, Ising, Sector, nactive
using Test

@testset "Test Sector System" begin
  @testset "U(1)" begin
    q0 = Sector(0)
    q1 = Sector(1)
    q2 = Sector(2)
    q3 = Sector(3)

    @test q0 ⊗ q0 == [q0]
    @test q0 ⊗ q1 == [q1]
    @test q1 ⊗ q0 == [q1]
    @test q0 ⊗ q2 == [q2]
    @test q1 ⊗ q2 == [q3]
  end

  @testset "Ƶ_2" begin
    z0 = Sector(0, Z(2))
    z1 = Sector(1, Z(2))

    @test z0 ⊗ z0 == [z0]
    @test z0 ⊗ z1 == [z1]
    @test z1 ⊗ z0 == [z1]
    @test z1 ⊗ z1 == [z0]
  end

  @testset "Ƶ_3" begin
    z0 = Sector(0, Z(3))
    z1 = Sector(1, Z(3))
    z2 = Sector(2, Z(3))

    @test z0 ⊗ z0 == [z0]
    @test z0 ⊗ z1 == [z1]
    @test z0 ⊗ z2 == [z2]
    @test z1 ⊗ z0 == [z1]
    @test z1 ⊗ z1 == [z2]
    @test z1 ⊗ z2 == [z0]
  end

  @testset "SU(2)" begin
    j0 = Sector(0, SU(2))
    j½ = Sector(1 / 2, SU(2))
    j1 = Sector(1, SU(2))
    j3_2 = Sector(3 / 2, SU(2))
    j2 = Sector(2, SU(2))

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
    d1 = Sector(1, SUd(2)) # spin 0
    d2 = Sector(2, SUd(2)) # spin 1/2
    d3 = Sector(3, SUd(2)) # spin 1
    d4 = Sector(4, SUd(2)) # spin 3/2
    d5 = Sector(5, SUd(2)) # spin 2

    @test d1 ⊗ d1 == [d1]
    @test d1 ⊗ d2 == [d2]
    @test d1 ⊗ d3 == [d3]

    @test d2 ⊗ d2 == d1 ⊕ d3
    @test d2 ⊗ d3 == d2 ⊕ d4
    @test d3 ⊗ d3 == d1 ⊕ d3 ⊕ d5
  end

  @testset "Ising" begin
    ı = Sector("1", Ising)
    σ = Sector("σ", Ising)
    ψ = Sector("ψ", Ising)

    @test ı ⊗ ı == [ı]
    @test ı ⊗ σ == [σ]
    @test ı ⊗ ψ == [ψ]
    @test σ ⊗ σ == ı ⊕ ψ
    @test ψ ⊗ σ == [σ]
    @test ψ ⊗ ψ == [ı]

    @test Sector(0, Ising) == ı
    @test Sector(1 / 2, Ising) == σ
    @test Sector(1, Ising) == ψ
  end

  @testset "Fibonacci" begin
    ı = Sector("1", Fib)
    τ = Sector("τ", Fib)

    @test ı ⊗ ı == [ı]
    @test ı ⊗ τ == [τ]
    @test τ ⊗ τ == ı ⊕ τ

    @test Sector(0, Fib) == ı
    @test Sector(1, Fib) == τ
  end

  @testset "SU(2) with z component" begin
    q½p = Sector("J", 1 / 2, 1 / 2, SUz(2))
    q½m = Sector("J", 1 / 2, -1 / 2, SUz(2))
    q1p = Sector("J", 1, +1, SUz(2))
    q10 = Sector("J", 1, 0, SUz(2))
    q1m = Sector("J", 1, -1, SUz(2))
    q00 = Sector("J", 0, 0, SUz(2))
    q22 = Sector("J", 2, +2, SUz(2))
    q21 = Sector("J", 2, +1, SUz(2))
    q20 = Sector("J", 2, 0, SUz(2))

    @test q½p ⊗ q½p == [q1p]
    @test q½p ⊗ q½m == q00 ⊕ q10
    @test q½m ⊗ q½m == [q1m]

    @test q1p ⊗ q1p == [q22]
    @test q1p ⊗ q10 == q1p ⊕ q21
    @test q1p ⊗ q1m == q00 ⊕ q10 ⊕ q20
  end

  @testset "Multiple U(1)'s" begin
    q00 = Sector()
    q10 = Sector("A", 1)
    q01 = Sector("B", 1)
    q11 = Sector(("A", 1), ("B", 1))

    @test q00 ⊗ q00 == [q00]
    @test q01 ⊗ q00 == [q01]
    @test q00 ⊗ q01 == [q01]
    @test q10 ⊗ q01 == [q11]
  end

  @testset "U(1) ⊗ SU(2)" begin
    q0 = Sector()
    q0h = Sector(("J", 1 / 2, SU(2)))
    q10 = Sector(("N", 1), ("J", 0, SU(2)))
    q1h = Sector(("N", 1), ("J", 1 / 2, SU(2)))
    q11 = Sector(("N", 1), ("J", 1, SU(2)))
    q20 = Sector(("N", 2))
    q2h = Sector(("N", 2), ("J", 1 / 2, SU(2)))
    q21 = Sector(("N", 2), ("J", 1, SU(2)))
    q22 = Sector(("N", 2), ("J", 2, SU(2)))

    @test q1h ⊗ q1h == q20 ⊕ q21
    @test q10 ⊗ q1h == [q2h]
    @test q0h ⊗ q1h == q10 ⊕ q11
    @test q11 ⊗ q11 == q20 ⊕ q21 ⊕ q22
  end

  @testset "Set all sectors" begin
    q = Sector(("A", 1))
    @test nactive(q) == 1
    q = Sector(("B", 2), ("A", 1))
    @test nactive(q) == 2
    q = Sector(("C", 3), ("B", 2), ("A", 1))
    @test nactive(q) == 3
    q = Sector(("D", 4), ("C", 3), ("B", 2), ("A", 1))
    @test nactive(q) == 4
  end

  @testset "Comparison with unspecified labels" begin
    q2 = Sector("N", 2, U(1))

    q20 = Sector(("N", 2, U(1)), ("J", 0, SU(2)))

    @test q20 == q2

    q21 = Sector(("N", 2, U(1)), ("J", 1, SU(2)))
    @test q21 != q2

    a = Sector(("A", 0), ("B", 2))
    b = Sector(("B", 2), ("C", 0))
    @test a == b
    c = Sector(("B", 2), ("C", 1))
    @test a != c
  end
end

nothing
