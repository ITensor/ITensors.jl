using ITensorGaussianMPS
using ITensors
using LinearAlgebra
using Test

function diagonalize_h(N)
  hterms = OpSum()
  for j in 1:(N - 1)
    hterms += -1, "Cdag", j, "C", j + 1
    hterms += -1, "Cdag", j + 1, "C", j
  end

  h = hopping_hamiltonian(hterms)
  ϵ, ϕ = eigen(h)
  return ϵ, ϕ
end

@testset "Green Function Properties" begin
  N = 20
  ϵ, ϕ = diagonalize_h(N)

  t = 1.0

  Gr = retarded_green_function(t, ϕ, ϵ)
  Gl = lesser_green_function(t, ϕ, ϵ)
  Gg = greater_green_function(t, ϕ, ϵ)

  @test Gr ≈ (Gg - Gl)

  # Gr is -im at time zero:
  @test retarded_green_function(0.0, ϕ, ϵ)[1, 1] ≈ -im

  # G< at t=0 is im times the correlation matrix
  Npart = div(N, 2)
  Gl0 = lesser_green_function(0, ϕ, ϵ; Npart)
  ϕ_occ = ϕ[:, 1:Npart]
  C = ϕ_occ * ϕ_occ'
  @test Gl0 ≈ im * C
end

@testset "Sites Keyword" begin
  N = 20
  ϵ, ϕ = diagonalize_h(N)

  t = 1.0

  G = retarded_green_function(t, ϕ, ϵ)

  G5_10 = retarded_green_function(t, ϕ, ϵ; sites=5:10)
  @test size(G5_10) == (6, 6)
  @test G5_10 ≈ G[5:10, 5:10]

  g7 = retarded_green_function(t, ϕ, ϵ; sites=7)
  @test g7 isa Number

  # Non-contiguous case
  Gnc = retarded_green_function(t, ϕ, ϵ; sites=[5, 10, 15])
  @test size(Gnc) == (3, 3)
  @test Gnc[1, 1] ≈ G[5, 5]
  @test Gnc[1, 2] ≈ G[5, 10]
  @test Gnc[2, 3] ≈ G[10, 15]
end

nothing
