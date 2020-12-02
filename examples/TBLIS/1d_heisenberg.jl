using ITensors
using LinearAlgebra
using Printf
using Random
using TBLIS

Random.seed!(1234)

let
  nthreads = 4

  N = 100
  sites = siteinds("S=1",N)
  ampo = AutoMPO()
  for j=1:N-1
    ampo += 0.5, "S+", j, "S-", j+1
    ampo += 0.5, "S-", j, "S+", j+1
    ampo +=      "Sz", j, "Sz", j+1
  end
  H = MPO(ampo,sites)
  psi0 = randomMPS(sites,10)

  sweeps_compile = Sweeps(2)
  maxdim!(sweeps_compile, 10)
  cutoff!(sweeps_compile, 1E-15)

  sweeps = Sweeps(6)
  maxdim!(sweeps, 20,100,200,300)
  cutoff!(sweeps, 0.0)
  @show sweeps

  #
  # Using BLAS backend
  #
 
  disable_tblis!()
  BLAS.set_num_threads(nthreads)

  # Compile
  dmrg(H, psi0, sweeps_compile; outputlevel = 0)

  println("Using BLAS with $nthreads threads\n")
  energy, psi = dmrg(H, psi0, sweeps)
  @printf("Final energy = %.12f\n",energy)
  println()

  #
  # Using TBLIS backend
  #

  enable_tblis!()
  BLAS.set_num_threads(1)
  TBLIS.set_num_threads(nthreads)

  # Compile
  dmrg(H, psi0, sweeps_compile; outputlevel = 0)

  println("Using TBLIS with $(TBLIS.get_num_threads()) threads (and 1 BLAS thread)\n")
  energy, psi = dmrg(H, psi0, sweeps)
  @printf("Final energy = %.12f\n",energy)
end

