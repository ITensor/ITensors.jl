import NDTensors.Sectors:
  ⊗,
  ⊕,
  dimension,
  Fib,
  fusion_rule,
  Ising,
  level,
  SU,
  trivial,
  U1,
  Z 
using Test

@testset "Test Category Types" begin

  @testset "U(1)" begin
    q1 = U1(1)
    q2 = U1(2)
    q3 = U1(3)

    @test dimension(q1) == 1
    @test dimension(q2) == 1

    @test q1 ⊗ q1 == [q2]
    @test q1 ⊗ q2 == [q3]
    @test q2 ⊗ q1 == [q3]

    @test trivial(U1) == U1(0)
  end

  @testset "Z₂" begin
    z0 = Z{2}(0)
    z1 = Z{2}(1)

    @test trivial(Z{2}) == Z{2}(0)

    @test dimension(z0) == 1
    @test dimension(z1) == 1

    @test z0 ⊗ z1 == [z1]
    @test z1 ⊗ z1 == [z0]
  end

  @testset "SU(2)" begin
    j1 = SU{2}(1)
    j2 = SU{2}(2)
    j3 = SU{2}(3)
    j4 = SU{2}(4)
    j5 = SU{2}(5)

    @test trivial(SU{2}) == SU{2}(1)

    @test dimension(j1) == 1
    @test dimension(j2) == 2
    @test dimension(j3) == 3
    @test dimension(j4) == 4
    
    @test j1 ⊗ j2 == [j2]
    @test j2 ⊗ j2 == j1 ⊕ j3
    @test j2 ⊗ j3 == j2 ⊕ j4
    @test j3 ⊗ j3 == j1 ⊕ j3 ⊕ j5

  end

  @testset "Fibonacci" begin
    ı = Fib("1")
    τ = Fib("τ")

    @test trivial(Fib) == ı

    @test ı ⊗ ı == [ı]
    @test ı ⊗ τ == [τ]
    @test τ ⊗ ı == [τ]
    @test τ ⊗ τ == ı ⊕ τ
  end

  @testset "Ising" begin
    ı = Ising("1")
    σ = Ising("σ")
    ψ = Ising("ψ")

    @test trivial(Ising) == ı

    @test ı ⊗ ı == [ı]
    @test ı ⊗ σ == [σ]
    @test σ ⊗ ı == [σ]
    @test ı ⊗ ψ == [ψ]
    @test ψ ⊗ ı == [ψ]
    @test σ ⊗ σ == ı ⊕ ψ
    @test σ ⊗ ψ == [σ]
    @test ψ ⊗ σ == [σ]
    @test ψ ⊗ ψ == [ı]
  end

end

nothing
