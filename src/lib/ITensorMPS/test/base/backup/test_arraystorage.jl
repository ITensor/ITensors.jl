using ITensors
using Test

@testset "Test ArrayStorage DMRG QN $conserve_qns" for conserve_qns in (false,) # true)
  n = 4
  s = siteinds("S=1/2", n; conserve_qns)
  heisenberg_opsum = function (n)
    os = OpSum()
    for j in 1:(n-1)
      os += "Sz", j, "Sz", j + 1
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
    end
    return os
  end
  H = MPO(heisenberg_opsum(n), s)
  ψ = random_mps(s, j -> isodd(j) ? "↑" : "↓"; linkdims=4)
  dmrg_kwargs = (; nsweeps=2, cutoff=[1e-4, 1e-12], maxdim=10, outputlevel=0)
  e1, ψ1 = dmrg(NDTensors.to_arraystorage.((H, ψ))...; dmrg_kwargs...)
  e2, ψ2 = dmrg(H, ψ; dmrg_kwargs...)
  @test e1 ≈ e2
end
