using ITensors,
      Printf

include("heisenberg.jl")

let
  N = 50

  sites = [Index(QN(1)=>1,QN(-1)=>1; tags="S=1/2,n=$n") for n in 1:N]

  #println("Using manual MPO")
  #H = heisenberg(sites)

  println("Using AutoMPO")
  ampo = AutoMPO()
  for j=1:N-1
    ampo += ("Sz",j,"Sz",j+1)
    ampo += (0.5,"S+",j,"S-",j+1)
    ampo += (0.5,"S-",j,"S+",j+1)
  end
  J2 = 0.1
  for j=1:N-2
    ampo += (J2,"Sz",j,"Sz",j+2)
  #  ampo += (0.5*J2,"S+",j,"S-",j+2)
  #  ampo += (0.5*J2,"S-",j,"S+",j+2)
  end
  H = toMPO(ampo,sites)

  psi0 = MPS(N)

  state = [isodd(n) ? 1 : 2 for n in 1:N] 

  l = [Index(QN()=>1; tags="l=$l") for l in 1:N-1]

  psi0 = productMPS(sites,state)

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

