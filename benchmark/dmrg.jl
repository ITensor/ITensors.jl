using ITensors,
      LinearAlgebra,
      Printf

function main()
  N = 100
  sites = spinOneSites(N)
  H = Heisenberg(sites)
  psi = randomMPS(sites)
  sw = Sweeps(5)
  maxdim!(sw,10,20,100,100,200)
  cutoff!(sw,1E-10)
  println("Starting DMRG")
  energy,psi = @time dmrg(H,psi,sw,maxiter=3)
  @printf "Final energy = %.12f\n" energy
end
main()
