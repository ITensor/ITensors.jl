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

  sites = siteinds("S=1",N,conserve_qns=true)

  ampo = AutoMPO()
  for j in 1:N-1
    ampo .+= 0.5, "S+", j, "S-", j+1
    ampo .+= 0.5, "S-", j, "S+", j+1
    ampo .+=      "Sz", j, "Sz", j+1
  end
  H = MPO(ampo, sites)

  # This step makes the MPO more sparse.
  # It generally improves DMRG performance
  # at large bond dimensions but makes DMRG slower at
  # small bond dimensions.
  H = splitblocks(linkinds, H)

  state = [isodd(n) ? "Up" : "Dn" for n in 1:N] 
  psi0 = randomMPS(sites,state,10)

  # Plan to do 5 DMRG sweeps:
  sweeps = Sweeps(5)
  # Set maximum MPS bond dimensions for each sweep
  setmaxdim!(sweeps, 10,20,100,100,200)
  # Set maximum truncation error allowed when adapting bond dimensions
  setcutoff!(sweeps, 1E-10)
  @show sweeps

  # Run the DMRG algorithm, returning energy and optimized MPS
  energy, psi = dmrg(H,psi0, sweeps)
  @printf("Final energy = %.12f\n",energy)
end

