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
