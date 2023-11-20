include("../../NDTensorsTestUtils/NDTensorsTestUtils.jl")
using .NDTensorsTestUtils: default_rtol, is_supported_eltype, devices_list
function test_dmrg(elt, N::Integer; dev::Function, conserve_qns)
  sites = siteinds("S=1/2", N; conserve_qns)

  os = OpSum()
  for j in 1:(N - 1)
    os += "Sz", j, "Sz", j + 1
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
  end

  Random.seed!(1234)
  init = j -> isodd(j) ? "↑" : "↓"
  psi0 = dev(randomMPS(elt, sites, init; linkdims=4))
  H = dev(MPO(elt, os, sites))

  nsweeps = 3
  cutoff = [1e-3, 1e-13]
  noise = [1e-6, 0]
  ## running these with nsweeps = 100 and no maxdim
  ## all problems do not have a maxlinkdim > 32
  maxdim = 32

  energy, psi = dmrg(H, psi0; nsweeps, cutoff, maxdim, noise, outputlevel=0)
  @test energy ≈ reference_energies[N] rtol = default_rtol(elt)
end
