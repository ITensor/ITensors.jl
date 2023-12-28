import NDTensors.Sectors: ⊗, ⊕, Fib, Ising, Sector, SU, SU2, U1, Z
using Test

@testset "Test Sector System" begin
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
    q10 = Sector(; N=U1(1), J=SU2(0))
    # Put names in reverse order sometimes:
    q1h = Sector(; J=SU2(1//2), N=U1(1))
    q11 = Sector(; N=U1(1), J=SU2(1))
    q20 = Sector(; N=U1(2))
    q2h = Sector(; N=U1(2), J=SU2(1//2))
    q21 = Sector(; N=U1(2), J=SU2(1))
    q22 = Sector(; N=U1(2), J=SU2(2))

    @test q1h ⊗ q1h == q20 ⊕ q21
    @test q10 ⊗ q1h == [q2h]
    @test q0h ⊗ q1h == q10 ⊕ q11
    @test q11 ⊗ q11 == q20 ⊕ q21 ⊕ q22
  end

  @testset "U(1) ⊗ SU(2)" begin
    q0 = Sector()
    q0h = Sector(; J=SU{2}(2))
    q10 = Sector(; N=U1(1), J=SU{2}(1))
    # Put names in reverse order sometimes:
    q1h = Sector(; J=SU{2}(2), N=U1(1))
    q11 = Sector(; N=U1(1), J=SU{2}(3))
    q20 = Sector(; N=U1(2))
    q2h = Sector(; N=U1(2), J=SU{2}(2))
    q21 = Sector(; N=U1(2), J=SU{2}(3))
    q22 = Sector(; N=U1(2), J=SU{2}(5))

    @test q1h ⊗ q1h == q20 ⊕ q21
    @test q10 ⊗ q1h == [q2h]
    @test q0h ⊗ q1h == q10 ⊕ q11
    @test q11 ⊗ q11 == q20 ⊕ q21 ⊕ q22
  end

  @testset "Comparisons with unspecified labels" begin
    q2 = Sector(; N=U1(2))
    q20 = Sector(; N=U1(2), J=SU{2}(1))
    @test q20 == q2

    q21 = Sector(; N=U1(2), J=SU{2}(3))
    @test q21 != q2

    a = Sector(; A=U1(0), B=U1(2))
    b = Sector(; B=U1(2), C=U1(0))
    @test a == b
    c = Sector(; B=U1(2), C=U1(1))
    @test a != c
  end
end

nothing
