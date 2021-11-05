using Test
using ITensors
using LinearAlgebra

@testset "Ops to MPO" begin
  ∑H = Ops.OpSum()
  ∑H += 1.2, "X", 1, "X", 2
  ∑H += 2, "Z", 1
  ∑H += 2, "Z", 2

  @test ∑H isa Ops.OpSum

  s = siteinds("Qubit", 2)
  H = MPO(∑H, s)

  Id(n) = Op(I, n)
  X(n) = Op("X", n)
  Z(n) = Op("Z", n)
  T(o) = ITensor(o, s)
  Hfull = 1.2 * T(X(1)) * T(X(2)) + 2 * T(Z(1)) * T(Id(2)) + 2 * T(Id(1)) * T(Z(2))

  @test prod(H) ≈ Hfull

  @test prod(MPO(X(1), s)) ≈ T(X(1)) * T(Id(2))
  @test prod(MPO(2X(1), s)) ≈ 2T(X(1)) * T(Id(2))
  @test prod(MPO(X(1) * Z(2), s)) ≈ T(X(1)) * T(Z(2))
  @test prod(MPO(3.5X(1) * Z(2), s)) ≈ 3.5T(X(1)) * T(Z(2))
  @test prod(MPO(X(1) + Z(2), s)) ≈ T(X(1)) * T(Id(2)) + T(Id(1)) * T(Z(2))
  @test prod(MPO(X(1) + 3.3Z(2), s)) ≈ T(X(1)) * T(Id(2)) + 3.3T(Id(1)) * T(Z(2))
  @test prod(MPO((X(1) + Z(2)) / 2, s)) ≈ 0.5T(X(1)) * T(Id(2)) + 0.5T(Id(1)) * T(Z(2))
end
