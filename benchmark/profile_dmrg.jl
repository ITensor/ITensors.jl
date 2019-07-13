using ITensors
using Printf
using Profile
using StatProfilerHTML

function main()
  N = 100
  sites = spinOneSites(N)
  H = Heisenberg(sites)
  psi = randomMPS(sites)
  sw = Sweeps(5)
  maxdim!(sw,10,20,100,100,200)
  cutoff!(sw,1E-11)

  smallsw = Sweeps(1)
  maxdim!(smallsw,10)
  cutoff!(smallsw,1E-4)
  @profile dmrg(H,psi,smallsw,maxiter=2)

  Profile.clear()
  @profile dmrg(H,psi,sw,maxiter=2)
  statprofilehtml()

  #energy,psi = @time dmrg(H,psi,sw,maxiter=2)
  #@printf "Final energy = %.12f\n" energy
  #@show inner(psi,H,psi)
end
main()
