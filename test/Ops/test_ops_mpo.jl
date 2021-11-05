using Test
using ITensors
using LinearAlgebra

@testset "Ops to MPO" begin
  ∑H = Ops.OpSum()
  ∑H += 1.2, "X", 1, "X", 2
  ∑H += 2, "Z", 1
  ∑H += 2, "Z", 2

  s = siteinds("Qubit", 2)
  H = MPO(∑H, s)
end
