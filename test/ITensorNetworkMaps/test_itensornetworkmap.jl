using Test
using ITensors
using ITensors.ITensorNetworkMaps
using KrylovKit: eigsolve
using LinearAlgebra

include(joinpath(@__DIR__, "utils", "utils.jl"))

@testset "ITensorNetworkMaps.jl" begin
  N = 3 # Number of sites in the unit cell
  e⃗ = [n => mod1(n + 1, N) for n in 1:N]
  χ⃗ = Dict()
  χ⃗[1 => 2] = 3
  χ⃗[2 => 3] = 4
  χ⃗[3 => 1] = 5
  d = 2
  ψ = infmps(N; χ⃗=χ⃗, d=d)
  randn!.(ψ.data)
  T = transfer_matrix(ψ)
  v = randomITensor(input_inds(T))
  Tv_expected = ITensors.contract([v, reverse(ψ)..., prime(linkinds, dag(reverse(ψ)))...])
  T′v_expected = ITensors.contract([v, ψ..., prime(linkinds, dag(ψ))...])
  @test T(v) ≈ Tv_expected
  @test (2T)(v) ≈ 2Tv_expected
  @test (2T + I)(v) ≈ 2Tv_expected + v
  @test (2T + 3I)(v) ≈ 2Tv_expected + 3v
  @test (T + T)(v) ≈ 2Tv_expected
  @test T'(v) ≈ T′v_expected
  @test (T + T')(v) ≈ Tv_expected + T′v_expected
  @test (T + T)'(v) ≈ 2T′v_expected
  @test all([T T; T T]([v; v]) .≈ [2Tv_expected; 2Tv_expected])

  T⃗ = transfer_matrices(ψ)

  @test (T⃗[1] * T⃗[2] * T⃗[3])(v) ≈ Tv_expected
  @test (T⃗[1] * T⃗[2] * T⃗[3] + 3I)(v) ≈ Tv_expected + 3v
  @test (T⃗[1] * T⃗[2] * T⃗[3])'(v) ≈ T′v_expected
  @test (I * T⃗[1] * T⃗[2] * T⃗[3])(v) ≈ Tv_expected
  @test (2T⃗[1] * T⃗[2] * T⃗[3])(v) ≈ 2Tv_expected
  @test (T⃗[1] * T⃗[2] * T⃗[3] + 2I)(v) ≈ Tv_expected + 2v
  @test ((T⃗[1] * T⃗[2] + T⃗[1] * T⃗[2]) * T⃗[3])(v) ≈ 2Tv_expected
  @test ((T⃗[1] * T⃗[2] + T⃗[1] * T⃗[2]) * T⃗[3])'(v) ≈ 2T′v_expected
  @test ((T⃗[1] * T⃗[2] + T⃗[1] * T⃗[2]) * T⃗[3] + 3I)(v) ≈ 2Tv_expected + 3v

  dk, vk = eigsolve(T, v)
  for n in 1:length(dk)
    @test norm((T - dk[n]I)(vk[n])) ≈ 0 atol = 1e-10
  end
end
