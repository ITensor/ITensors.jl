using ITensors
using Test
using Random

Random.seed!(12345)

include(joinpath(pkgdir(ITensors), "examples", "src", "trg.jl"))
include(joinpath(pkgdir(ITensors), "examples", "src", "2d_classical_ising.jl"))

@testset "trg" begin
  # Make Ising model partition function
  β = 1.1 * βc
  d = 2
  s = Index(d)
  l = addtags(s, "left")
  u = addtags(s, "up")
  T = ising_mpo(l, u, β)

  χmax = 20
  nsteps = 20
  κ, T = trg(T; χmax=χmax, nsteps=nsteps)

  @test κ ≈ exp(-β * ising_free_energy(β)) atol = 1e-4
end
