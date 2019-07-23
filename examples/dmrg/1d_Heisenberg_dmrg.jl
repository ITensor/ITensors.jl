using ITensors
using Printf

# Use DMRG to solve the spin 1, 1D Heisenberg model on 100 sites
# For the Heisenberg model in one dimension
# H = J ∑ᵢ(SᶻᵢSᶻᵢ₊₁ + SˣᵢSˣᵢ₊₁ + SʸᵢSʸᵢ₊₁ )
#   = J ∑ᵢ[SᶻᵢSᶻᵢ₊₁ + ½(S⁺ᵢS⁻ᵢ₊₁ + S⁻ᵢS⁺ᵢ₊₁)]
let
  N = 100                             # Number of sites
  sites = spinOneSites(N)             # Create a 1D, N site lattice of spin 1 degrees of freedom

  ampo = AutoMPO(sites)               # Initialize an MPO on the lattice
  for j=1:N-1
      add!(ampo,"Sz",j,"Sz",j+1)      # Put in nearest neighbor SᶻSᶻ  term
      add!(ampo,0.5,"S+",j,"S-",j+1)  # Put in nearest neighbor ½S⁺S⁻ term
      add!(ampo,0.5,"S-",j,"S+",j+1)  # Put in nearest neighbor ½S⁻S⁺ term
  end
  H = toMPO(ampo)                     # Create Hamiltonian MPO

  ψ₀ = randomMPS(sites)               # Initalize a random matrix product state as the DMRG starting point

  sweeps = Sweeps(5)                  # Do 5 DMRG sweeps
  maxdim!(sweeps, 10,20,100,100,200)  # Set the maximum bond dimensions for each sweep
  cutoff!(sweeps, 1E-10)              # Set the maximum truncation error allowed when computing SVD or matrix diagonalizations
  @show sweeps               

  energy, psi = dmrg(H, ψ₀, sweeps)   # Perform the variational DMRG algorithm to find the ground state and ground state energy of H using psi0 and sweeps. 
  @printf("Final energy = %.12f\n",energy)
end
