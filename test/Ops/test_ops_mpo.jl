using Test
using ITensors
using ITensors.Ops
using LinearAlgebra

@testset "Ops to MPO" begin
  ∑H = Sum{Op}()
  ∑H += 1.2, "X", 1, "X", 2
  ∑H += 2, "Z", 1
  ∑H += 2, "Z", 2

  @test ∑H isa Sum{Scaled{Float64,Prod{Op}}}

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

  @testset "OpSum to MPO with repeated terms" begin
    ℋ = OpSum()
    ℋ += "Z", 1
    ℋ += "Z", 1
    ℋ += "X", 2
    ℋ += "Z", 1
    ℋ += "Z", 1
    ℋ += "X", 2
    ℋ += "X", 2
    ℋ_merged = OpSum()
    ℋ_merged += (4, "Z", 1)
    ℋ_merged += (3, "X", 2)
    @test ITensors.sortmergeterms(ℋ) == ℋ_merged

    # Test with repeated terms
    s = siteinds("S=1/2", 1)
    ℋ = OpSum() + ("Z", 1) + ("Z", 1)
    H = MPO(ℋ, s)
    @test contract(H) ≈ 2 * op("Z", s, 1)
  end
end

function heisenberg_old(N)
  os = OpSum()
  for j in 1:(N - 1)
    os += "Sz", j, "Sz", j + 1
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
  end
  return os
end

function heisenberg(N)
  os = Sum{Op}()
  for j in 1:(N - 1)
    os += "Sz", j, "Sz", j + 1
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
  end
  return os
end

@testset "OpSum comparison" begin
  N = 4
  s = siteinds("S=1/2", N)
  os_old = heisenberg_old(N)
  os_new = heisenberg(N)
  @test os_old isa OpSum
  @test os_new isa Sum{Scaled{Float64,Prod{Op}}}
  Hold = MPO(os_old, s)
  Hnew = MPO(os_new, s)
  @test prod(Hold) ≈ prod(Hnew)
end

@testset "Square Hamiltonian" begin
  N = 4
  ℋ = heisenberg(N)
  ℋ² = expand(ℋ^2)
  s = siteinds("S=1/2", N)
  H = MPO(ℋ, s)
  H² = MPO(ℋ², s)
  @test norm(replaceprime(H' * H, 2 => 1) - H²) ≈ 0 atol = 1e-14
  @test norm(H(H) - H²) ≈ 0 atol = 1e-14
end
