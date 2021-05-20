using ITensors

#
# DMRG calculation of the extended Hubbard model
# ground state wavefunction, and spin densities
#

# Usage:
#
# Run:
#
# $ julia exthubbard.jl input.jl
#

# Include the specified input file
# Include the file from the first positional
# argument. If none is specified, use
# input.jl by default.
include(get(ARGS, 1, "input.jl"))

sweeps = Sweeps(nsweep, sweeps_args)

@show sweeps
@show N, Npart

sites = siteinds("Electron", N; conserve_qns=true)

ampo = OpSum()
for i in 1:N
  ampo .+= U, "Nupdn", i
end
for b in 1:(N - 1)
  ampo .+= -t1, "Cdagup", b, "Cup", b + 1
  ampo .+= -t1, "Cdagup", b + 1, "Cup", b
  ampo .+= -t1, "Cdagdn", b, "Cdn", b + 1
  ampo .+= -t1, "Cdagdn", b + 1, "Cdn", b
  ampo .+= V1, "Ntot", b, "Ntot", b + 1
end
for b in 1:(N - 2)
  ampo .+= -t2, "Cdagup", b, "Cup", b + 2
  ampo .+= -t2, "Cdagup", b + 2, "Cup", b
  ampo .+= -t2, "Cdagdn", b, "Cdn", b + 2
  ampo .+= -t2, "Cdagdn", b + 2, "Cdn", b
end
H = MPO(ampo, sites)

state = ["Emp" for n in 1:N]
p = Ref(Npart)
for i in N:-1:1
  if p[] > i
    println("Doubly occupying site $i")
    state[i] = "UpDn"
    p[] -= 2
  elseif p[] > 0
    println("Singly occupying site $i")
    state[i] = (isodd(i) ? "Up" : "Dn")
    p[] -= 1
  end
end
# Initialize wavefunction to be bond 
# dimension 10 random MPS with number
# of particles the same as `state`
psi0 = randomMPS(sites, state, 10)

# Check total number of particles:
@show flux(psi0)

# Start DMRG calculation:
energy, psi = dmrg(H, psi0, sweeps)

upd = fill(0.0, N)
dnd = fill(0.0, N)
for j in 1:N
  orthogonalize!(psi, j)
  psidag_j = dag(prime(psi[j], "Site"))
  upd[j] = scalar(psidag_j * op(sites, "Nup", j) * psi[j])
  dnd[j] = scalar(psidag_j * op(sites, "Ndn", j) * psi[j])
end

println("Up Density:")
for j in 1:N
  println("$j $(upd[j])")
end
println()

println("Dn Density:")
for j in 1:N
  println("$j $(dnd[j])")
end
println()

println("Total Density:")
for j in 1:N
  println("$j $(upd[j]+dnd[j])")
end
println()

println("\nGround State Energy = $energy")
