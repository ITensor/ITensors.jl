using ITensors
using Printf
using Random

# Use DMRG to solve the spin 1, 1D Heisenberg model on 100 sites
# For the Heisenberg model in one dimension
# H = J ∑ᵢ(SᶻᵢSᶻᵢ₊₁ + SˣᵢSˣᵢ₊₁ + SʸᵢSʸᵢ₊₁ )
#   = J ∑ᵢ[SᶻᵢSᶻᵢ₊₁ + ½(S⁺ᵢS⁻ᵢ₊₁ + S⁻ᵢS⁺ᵢ₊₁)]
# We'll work in units where J=1

let
  N = 100

  # Create N spin-one degrees of freedom
  sites = siteinds("S=1",N)
  # Alternatively can make spin-half sites instead
  #sites = siteinds("S=1/2",N)

  # Input operator terms which define a Hamiltonian
  ampo = AutoMPO()
  for j=1:N-1
      add!(ampo,"Sz",j,"Sz",j+1)
      add!(ampo,0.5,"S+",j,"S-",j+1)
      add!(ampo,0.5,"S-",j,"S+",j+1)
  end
  # Convert these terms to an MPO tensor network
  H = toMPO(ampo,sites)

  # Create an initial random matrix product state
  psi0 = randomMPS(sites)

  # Plan to do 5 DMRG sweeps:
  sweeps = Sweeps(5)
  # Set maximum MPS bond dimensions for each sweep
  maxdim!(sweeps, 10,20,100,100,200)
  # Set maximum truncation error allowed when adapting bond dimensions
  cutoff!(sweeps, 1E-10)
  @show sweeps

  # Run the DMRG algorithm, returning energy and optimized MPS
  energy, psi = dmrg(H,psi0, sweeps)
  @printf("Final energy = %.12f\n",energy)
end

