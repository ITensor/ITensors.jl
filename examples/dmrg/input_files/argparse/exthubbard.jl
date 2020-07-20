using ITensors

#
# DMRG calculation of the extended Hubbard model
# ground state wavefunction, and spin densities
#

# Usage: 
#
# First install `ArgParse.jl`:
#
# julia>] add ArgParse
#
# Then run:
#
# $ julia exthubbard.jl input.jl
#
# or with options, like
#
# $ julia exthubbard.jl input.jl --N=100 --Npart=20 --nsweep=4 --cutoff=1e-11 --maxdim=20 40 80 100 --noise=1e-12 1e-13
#
# In the REPL (if you just type `julia`), you can call it as follows:
#
# julia> push!(ARGS, "input.jl");
#
# julia> push!(ARGS, "--N=100", "--Npart=20");
#
# julia> include("exthubbard.jl");
#

let
  # Include the specified input file
  isempty(ARGS) && error("""Must specify input file, for example with `julia exthubbard.jl input.jl` or interactively with `push!(ARGS, "input.jl")`""")
  include(ARGS[1])

  sweeps = Sweeps(nsweep)
  maxdim!(sweeps, maxdim...)
  mindim!(sweeps, mindim...)
  cutoff!(sweeps, cutoff...)
  noise!(sweeps, noise...)

  @show sweeps
  @show N, Npart

  sites = siteinds("Electron",N; conserve_qns=true)

  ampo = AutoMPO()
  for i=1:N
    ampo += U,"Nupdn",i
  end
  for b=1:N-1
    ampo += -t1,"Cdagup",b,"Cup",b+1
    ampo += -t1,"Cdagup",b+1,"Cup",b
    ampo += -t1,"Cdagdn",b,"Cdn",b+1
    ampo += -t1,"Cdagdn",b+1,"Cdn",b
    ampo += V1,"Ntot",b,"Ntot",b+1
  end
  for b=1:N-2
    ampo += -t2,"Cdagup",b,"Cup",b+2
    ampo += -t2,"Cdagup",b+2,"Cup",b
    ampo += -t2,"Cdagdn",b,"Cdn",b+2
    ampo += -t2,"Cdagdn",b+2,"Cdn",b
  end
  H = MPO(ampo,sites)

  state = ["Emp" for n=1:N]
  p = Npart
  for i=N:-1:1
    if p > i
      println("Doubly occupying site $i")
      state[i] = "UpDn"
      p -= 2
    elseif p > 0
      println("Singly occupying site $i")
      state[i] = (isodd(i) ? "Up" : "Dn")
      p -= 1
    end
  end
  # Initialize wavefunction to be bond 
  # dimension 10 random MPS with number
  # of particles the same as `state`
  psi0 = randomMPS(sites,state,10)

  # Check total number of particles:
  @show flux(psi0)

  # Start DMRG calculation:
  energy,psi = dmrg(H,psi0,sweeps)

  upd = fill(0.0,N)
  dnd = fill(0.0,N)
  for j=1:N
    orthogonalize!(psi,j)
    psidag_j = dag(prime(psi[j], "Site"))
    upd[j] = scalar(psidag_j * op(sites, "Nup", j) * psi[j])
    dnd[j] = scalar(psidag_j * op(sites, "Ndn", j) * psi[j])
  end

  println("Up Density:")
  for j=1:N
    println("$j $(upd[j])")
  end
  println()

  println("Dn Density:")
  for j=1:N
    println("$j $(dnd[j])")
  end
  println()

  println("Total Density:")
  for j=1:N
    println("$j $(upd[j]+dnd[j])")
  end
  println()

  println("\nGround State Energy = $energy")

end
