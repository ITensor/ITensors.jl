using ITensors
using ITensors.NDTensors
using Test

@testset "dmrg" begin
  N = 10
  sites = siteinds("S=1", N)
  opsum = OpSum()
  for j in 1:(N - 1)
    opsum += "Sz", j, "Sz", j + 1
    opsum += 0.5, "S+", j, "S-", j + 1
    opsum += 0.5, "S-", j, "S+", j + 1
  end
  H = MPO(opsum, sites)
  psi0 = randomMPS(sites; linkdims=10)
  sweeps = Sweeps(5)
  setmaxdim!(sweeps, 10, 20, 100, 100, 200)
  setcutoff!(sweeps, 1E-11)
  @test @inferred(Tuple{Any,MPS}, dmrg(H, psi0, sweeps; outputlevel=0)) isa
    Tuple{Float64,MPS}
end
