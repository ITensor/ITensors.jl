using Test
using ITensors
using NDTensors
using Random 

Random.seed!(1234)

function test_dmrg(elt, N::Integer, dev::Function)
  sites = siteinds("S=1/2", N)

  os = OpSum()
  for j in 1:(N - 1)
    os += "Sz", j, "Sz", j + 1
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
  end

  H = dev(MPO(elt, os, sites; linkdims=4))

  psi0 = dev(randomMPS(elt, sites;))

  nsweeps = 7
  cutoff = [1e-3, 1e-13]
  noise = [1e-12, 0]

  energy, psi = dmrg(H, psi0; nsweeps, cutoff, noise, outputlevel=0)
  @test energy ≈ get_ref_value(dev, N, elt)
end
