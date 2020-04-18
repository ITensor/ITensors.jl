using ITensors,
      Printf

let
  N = 100

  sites = siteinds("S=1",N,conserve_qns=true)

  ampo = AutoMPO()
  for j=1:N-1
    ampo += ("Sz",j,"Sz",j+1)
    ampo += (0.5,"S+",j,"S-",j+1)
    ampo += (0.5,"S-",j,"S+",j+1)
  end
  H = toMPO(ampo,sites)

  psi0 = MPS(N)

  state = [isodd(n) ? "Up" : "Dn" for n in 1:N] 
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

