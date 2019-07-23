using ITensors
using Printf
#using Logging
##Set global logging level to Debug to show timers
#global_logger(SimpleLogger(stdout, Logging.Debug))

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

  sweeps = Sweeps(1)
  maxdim!(sweeps,10,20,100,100,200)
  cutoff!(sweeps,1E-11)
  @show sweeps

  energy,psi = @time dmrg(H,psi,sweeps,maxiter=2)
  @printf "Final energy = %.12f\n" energy
end
main()
