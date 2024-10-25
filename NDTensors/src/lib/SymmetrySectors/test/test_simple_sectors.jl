@eval module $(gensym())
using NDTensors.GradedAxes: dual
using NDTensors.SymmetrySectors:
  Fib,
  Ising,
  O2,
  SU,
  SU2,
  TrivialSector,
  U1,
  Z,
  quantum_dimension,
  fundamental,
  istrivial,
  trivial
using Test: @inferred, @test, @testset, @test_throws
@testset "Test SymmetrySectors Types" begin
  @testset "TrivialSector" begin
    q = TrivialSector()

    @test (@inferred quantum_dimension(q)) == 1
    @test q == q
    @test trivial(q) == q
    @test istrivial(q)

    @test dual(q) == q
    @test !isless(q, q)
  end

  @testset "U(1)" begin
    q1 = U1(1)
    q2 = U1(2)
    q3 = U1(3)

    @test quantum_dimension(q1) == 1
    @test quantum_dimension(q2) == 1
    @test (@inferred quantum_dimension(q1)) == 1

    @test trivial(q1) == U1(0)
    @test trivial(U1) == U1(0)
    @test istrivial(U1(0))

    @test dual(U1(2)) == U1(-2)
    @test isless(U1(1), U1(2))
    @test !isless(U1(2), U1(1))
    @test U1(Int8(1)) == U1(1)
    @test U1(UInt32(1)) == U1(1)

    @test U1(0) == TrivialSector()
    @test TrivialSector() == U1(0)
    @test U1(-1) < TrivialSector()
    @test TrivialSector() < U1(1)
    @test U1(Int8(1)) < U1(Int32(2))
  end

  @testset "Z₂" begin
    z0 = Z{2}(0)
    z1 = Z{2}(1)

    @test trivial(Z{2}) == Z{2}(0)
    @test istrivial(Z{2}(0))

    @test quantum_dimension(z0) == 1
    @test quantum_dimension(z1) == 1
    @test (@inferred quantum_dimension(z0)) == 1

    @test dual(z0) == z0
    @test dual(z1) == z1

    @test dual(Z{2}(1)) == Z{2}(1)
    @test isless(Z{2}(0), Z{2}(1))
    @test !isless(Z{2}(1), Z{2}(0))
    @test Z{2}(0) == z0
    @test Z{2}(-3) == z1

    @test Z{2}(0) == TrivialSector()
    @test TrivialSector() < Z{2}(1)
    @test_throws MethodError U1(0) < Z{2}(1)
    @test Z{2}(0) != Z{2}(1)
    @test Z{2}(0) != Z{3}(0)
    @test Z{2}(0) != U1(0)
  end

  @testset "O(2)" begin
    s0e = O2(0)
    s0o = O2(-1)
    s12 = O2(1//2)
    s1 = O2(1)

    @test trivial(O2) == s0e
    @test istrivial(s0e)

    @test (@inferred quantum_dimension(s0e)) == 1
    @test (@inferred quantum_dimension(s0o)) == 1
    @test (@inferred quantum_dimension(s12)) == 2
    @test (@inferred quantum_dimension(s1)) == 2

    @test (@inferred dual(s0e)) == s0e
    @test (@inferred dual(s0o)) == s0o
    @test (@inferred dual(s12)) == s12
    @test (@inferred dual(s1)) == s1

    @test s0o < s0e < s12 < s1
    @test s0e == TrivialSector()
    @test s0o < TrivialSector()
    @test TrivialSector() < s12
  end

  @testset "SU(2)" begin
    j1 = SU2(0)
    j2 = SU2(1//2)  # Rational will be cast to HalfInteger
    j3 = SU2(1)
    j4 = SU2(3//2)

    # alternative constructors
    @test j2 == SU{2}((1,))  # tuple SU(N)-like constructor
    @test j2 == SU{2,1}((1,))  # tuple constructor with explicit {N,N-1}
    @test j2 == SU((1,))  # infer N from tuple length
    @test j2 == SU{2}((Int8(1),))  # any Integer type accepted
    @test j2 == SU{2}((UInt32(1),))  # any Integer type accepted
    @test j2 == SU2(1 / 2)  # Float will be cast to HalfInteger
    @test_throws MethodError SU2((1,))  # avoid confusion between tuple and half-integer interfaces
    @test_throws MethodError SU{2,1}(1)  # avoid confusion

    @test trivial(SU{2}) == SU2(0)
    @test istrivial(SU2(0))
    @test fundamental(SU{2}) == SU2(1//2)

    @test quantum_dimension(j1) == 1
    @test quantum_dimension(j2) == 2
    @test quantum_dimension(j3) == 3
    @test quantum_dimension(j4) == 4
    @test (@inferred quantum_dimension(j1)) == 1

    @test dual(j1) == j1
    @test dual(j2) == j2
    @test dual(j3) == j3
    @test dual(j4) == j4

    @test j1 < j2 < j3 < j4
    @test SU2(0) == TrivialSector()
    @test !(j2 < TrivialSector())
    @test TrivialSector() < j2
  end

  @testset "SU(N)" begin
    f3 = SU{3}((1, 0))
    f4 = SU{4}((1, 0, 0))
    ad3 = SU{3}((2, 1))
    ad4 = SU{4}((2, 1, 1))

    @test trivial(SU{3}) == SU{3}((0, 0))
    @test istrivial(SU{3}((0, 0)))
    @test trivial(SU{4}) == SU{4}((0, 0, 0))
    @test istrivial(SU{4}((0, 0, 0)))
    @test SU{3}((0, 0)) == TrivialSector()
    @test SU{4}((0, 0, 0)) == TrivialSector()

    @test fundamental(SU{3}) == f3
    @test fundamental(SU{4}) == f4

    @test dual(f3) == SU{3}((1, 1))
    @test dual(f4) == SU{4}((1, 1, 1))
    @test dual(ad3) == ad3
    @test dual(ad4) == ad4

    @test quantum_dimension(f3) == 3
    @test quantum_dimension(f4) == 4
    @test quantum_dimension(ad3) == 8
    @test quantum_dimension(ad4) == 15
    @test quantum_dimension(SU{3}((4, 2))) == 27
    @test quantum_dimension(SU{3}((3, 3))) == 10
    @test quantum_dimension(SU{3}((3, 0))) == 10
    @test quantum_dimension(SU{3}((0, 0))) == 1
    @test (@inferred quantum_dimension(f3)) == 3
  end

  @testset "Fibonacci" begin
    ı = Fib("1")
    τ = Fib("τ")

    @test trivial(Fib) == ı
    @test istrivial(ı)
    @test ı == TrivialSector()

    @test dual(ı) == ı
    @test dual(τ) == τ

    @test (@inferred quantum_dimension(ı)) == 1.0
    @test (@inferred quantum_dimension(τ)) == ((1 + √5) / 2)

    @test ı < τ
  end

  @testset "Ising" begin
    ı = Ising("1")
    σ = Ising("σ")
    ψ = Ising("ψ")

    @test trivial(Ising) == ı
    @test istrivial(ı)
    @test ı == TrivialSector()

    @test dual(ı) == ı
    @test dual(σ) == σ
    @test dual(ψ) == ψ

    @test (@inferred quantum_dimension(ı)) == 1.0
    @test (@inferred quantum_dimension(σ)) == √2
    @test (@inferred quantum_dimension(ψ)) == 1.0

    @test ı < σ < ψ
  end
end
end
