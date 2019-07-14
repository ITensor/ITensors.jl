using ITensors
using Printf

function main()
  N = 100
  sites = spinOneSites(N)

  ampo = AutoMPO(sites)
  for j=1:N-1
    add!(ampo,"Sz",j,"Sz",j+1)
    add!(ampo,0.5,"S+",j,"S-",j+1)
    add!(ampo,0.5,"S-",j,"S+",j+1)
  end
  H = toMPO(ampo)

  psi = randomMPS(sites)

  sw = Sweeps(5)
  maxdim!(sw,10,20,100,100,200)
  cutoff!(sw,1E-11)

  println("Starting DMRG")
  energy,psi = @time dmrg(H,psi,sw,maxiter=2)
  @printf "Final energy = %.12f\n" energy
  @show inner(psi,H,psi)
end
main()
