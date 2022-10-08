using ITensors
using Printf
using Random

Random.seed!(1234)

let
  N = 100

  # Create N spin-one degrees of freedom
  sites = siteinds("S=1", N)
  # Alternatively can make spin-half sites instead
  #sites = siteinds("S=1/2",N)

  # Input operator terms which define a Hamiltonian
  os = OpSum()
  for j in 1:(N - 1)
    os += "Sz", j, "Sz", j + 1
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
  end
  # Convert these terms to an MPO tensor network
  H = MPO(os, sites)

  # Create an initial random matrix product state
  psi0 = randomMPS(sites, 10)

  # Plan to do 5 DMRG sweeps:
  nsweeps = 5
  # Set maximum MPS bond dimensions for each sweep
  maxdim = [10, 20, 100, 100, 200]
  # Set maximum truncation error allowed when adapting bond dimensions
  cutoff = [1E-11]

  # Run the DMRG algorithm, returning energy and optimized MPS
  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
  @printf("Final energy = %.12f\n", energy)
end
