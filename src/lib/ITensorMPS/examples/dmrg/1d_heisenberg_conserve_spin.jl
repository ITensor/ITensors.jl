using ITensors
using ITensors.Strided
using LinearAlgebra
using Printf
using Random

Random.seed!(1234)
BLAS.set_num_threads(1)
Strided.set_num_threads(1)
ITensors.enable_threaded_blocksparse()
#ITensors.disable_threaded_blocksparse()

let
  N = 100

  sites = siteinds("S=1", N; conserve_qns=true)

  os = OpSum()
  for j in 1:(N - 1)
    os .+= 0.5, "S+", j, "S-", j + 1
    os .+= 0.5, "S-", j, "S+", j + 1
    os .+= "Sz", j, "Sz", j + 1
  end
  H = MPO(os, sites)

  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  psi0 = randomMPS(sites, state, 10)

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
