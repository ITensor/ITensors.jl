using ITensorGaussianMPS
using ITensors
using LinearAlgebra
using Test

function diagonalize_h(N)
  hterms = OpSum()
  for j=1:N-1
    hterms += -1,"Cdag",j,"C",j+1
    hterms += -1,"Cdag",j+1,"C",j
  end

  h = hopping_hamiltonian(hterms)
  ϵ, phi = eigen(h)
  return ϵ,phi
end

@testset "Green Function Properties" begin
  N = 20
  ϵ,phi = diagonalize_h(N)

  t = 1.0

  Gr = G_R(t,phi,ϵ)
  Gl = G_L(t,phi,ϵ)
  Gg  = G_G(t,phi,ϵ)

  @test Gr ≈ (Gg-Gl)

  # Gr is -i at time zero:
  @test G_R(0.0,phi,ϵ)[1,1] ≈ -1.0im
end

@testset "Sites Keyword" begin
  N = 20
  ϵ,phi = diagonalize_h(N)

  t = 1.0

  G = G_R(t,phi,ϵ)

  G5_10 = G_R(t,phi,ϵ; sites=5:10)
  @test size(G5_10) == (6,6)
  @test G5_10 ≈ G[5:10,5:10]

  g = G_R(t,phi,ϵ; sites=7)
  @test g isa Number

  # Non-contiguous case
  Gnc = G_R(t,phi,ϵ; sites=[5,10,15])
  @test size(Gnc) == (3,3)
end


nothing
