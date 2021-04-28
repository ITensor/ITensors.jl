using ITensors
using Test
import Random: seed!

seed!(12345)

include(joinpath(@__DIR__, "..", "examples", "src", "trg.jl"))
include(joinpath(@__DIR__, "..", "examples", "src", "2d_classical_ising.jl"))

@testset "trg" begin
  # Make Ising model MPO
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

nothing
