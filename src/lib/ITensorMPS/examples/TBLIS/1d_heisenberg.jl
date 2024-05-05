using ITensors
using LinearAlgebra
using Printf
using Random
using TBLIS

Random.seed!(1234)

let
  nthreads = 4

  N = 100
  sites = siteinds("S=1", N)
  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  H = MPO(os, sites)
  psi0 = randomMPS(sites, 10)

  nsweeps = 6
  maxdim = [20, 100, 200, 300]
  cutoff = 1E-15

  #
  # Using BLAS backend
  #

  ITensors.disable_tblis()
  BLAS.set_num_threads(nthreads)

  # Compile
  dmrg(H, psi0; nsweeps=2, maxdim=10, outputlevel=0)

  println("Using BLAS with $nthreads threads\n")
  energy, psi = @time dmrg(H, psi0; nsweeps, maxdim, cutoff)
  @printf("Final energy = %.12f\n", energy)
  println()

  #
  # Using TBLIS backend
  #

  ITensors.enable_tblis()
  BLAS.set_num_threads(1)
  TBLIS.set_num_threads(nthreads)

  # Compile
  dmrg(H, psi0; nsweeps=2, maxdim=10, outputlevel=0)

  println("Using TBLIS with $(TBLIS.get_num_threads()) threads (and 1 BLAS thread)\n")
  energy, psi = @time dmrg(H, psi0; nsweeps, maxdim, cutoff)
  @printf("Final energy = %.12f\n", energy)
end
