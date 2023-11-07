using Test
using ITensors
using NDTensors
using Random

function test_dmrg(elt, N::Integer, dev::Function)
  sites = siteinds("S=1/2", N)

  os = OpSum()
  for j in 1:(N - 1)
    os += "Sz", j, "Sz", j + 1
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
  end

  Random.seed!(1234)
  psi0 = dev(randomMPS(Float64, sites;))
  H = dev(MPO(elt, os, sites; linkdims=4))

  nsweeps = 3
  cutoff = [1e-3, 1e-13]
  noise = [1e-12, 0]
  ## running these with nsweeps = 100 and no maxdim
  ## all problems do not have a maxlinkdim > 32
  maxdim = 32

  energy, psi = dmrg(H, psi0; nsweeps, cutoff, maxdim, noise, outputlevel=0)
  @test energy ≈ reference_energies[N] rtol = eps(real(elt)) * 10^4
end
