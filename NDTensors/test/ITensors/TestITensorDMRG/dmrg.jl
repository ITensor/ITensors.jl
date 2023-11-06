using Test
using ITensors
using NDTensors

function test_dmrg(
  elt,
  N::Integer, 
  dev::Function, 
  cut::Float64,
  no::Float64)
  # Create N spin-one degrees of freedom
  sites = siteinds("S=1", N)

  # Input operator terms which define a Hamiltonian
  os = OpSum()
  for j in 1:(N - 1)
    os += "Sz", j, "Sz", j + 1
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
  end

  # Convert these terms to an MPO tensor network
  H = dev(MPO(elt, os, sites))

  # Create an initial random matrix product state
  psi0 = dev(randomMPS(elt, sites;))

  # Plan to do 5 DMRG sweeps:
  nsweeps = 3
  # Set maximum MPS bond dimensions for each sweep
  maxdim = [10, 20, 100, 100, 200]
  # Set maximum truncation error allowed when adapting bond dimensions
  cutoff = [cut]
  # Set the noise
  noise = [no]

  # Run the DMRG algorithm, returning energy and optimized MPS
  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, noise, outputlevel=0)
  ref = TestITensorDMRG.get_ref_value(dev,N,cut,no,elt)
  println("$N, $cut, $no, $elt, $energy, $ref")
  @test energy ≈ get_ref_value(dev, N, cut, no, elt)
end


