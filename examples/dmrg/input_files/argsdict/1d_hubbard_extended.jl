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
# or with options, like
#
# $ julia exthubbard.jl input.jl N=100 Npart=20 nsweep=4 cutoff=1e-11 maxdim=20,40,80,100 noise=1e-12,1e-13
#
# In the REPL (if you just type `julia`), you can modify the command line options as follows (or just modify "input.jl" and reinclude it):
#
# julia> push!(ARGS, "input.jl");
#
# julia> push!(ARGS, "N=100", "Npart=20");
#
# julia> include("exthubbard.jl");
#

# Parse arguments to overide defaults
args = argsdict(; prefix="file")

# Include the file from the first positional
# argument. If none is specified, use
# input.jl by default.
include(get(args, "file1", "input.jl"))

# Create the sweeps object from the input
# file
sweeps = Sweeps(nsw, sweeps_args)
# Extract the original (default) values
maxdim = get_maxdims(sweeps)
mindim = get_mindims(sweeps)
cutoff = get_cutoffs(sweeps)
noise = get_noises(sweeps)

# Extract the argument values from
# the dictionary

N = get(args, "N", N)
Npart = get(args, "Npart", Npart)
t1 = get(args, "t1", t1)
t2 = get(args, "t2", t2)
U = get(args, "U", U)
V1 = get(args, "V1", V1)
nsw = get(args, "nsweep", nsw)
maxdim = get(args, "maxdim", maxdim)
mindim = get(args, "mindim", mindim)
cutoff = get(args, "cutoff", cutoff)
noise = get(args, "noise", noise)

#
# Alternatively, define all of the
# variables for all of the inputs
# programatically. You can use the
# keyword argument `as_symbols = true`
# in the `parse_args` function to
# automatically turn the keys of
# the dictionary into symbols.
#

#for (arg, val) in args
#  arg == "input_file" && continue
#  @eval $(Symbol(arg)) = $val
#end

sweeps = Sweeps(nsw)
maxdim!(sweeps, maxdim...)
mindim!(sweeps, mindim...)
cutoff!(sweeps, cutoff...)
noise!(sweeps, noise...)

@show sweeps
@show N, Npart

sites = siteinds("Electron", N; conserve_qns=true)

os = OpSum()
for i in 1:N
  os .+= U, "Nupdn", i
end
for b in 1:(N - 1)
  os .+= -t1, "Cdagup", b, "Cup", b + 1
  os .+= -t1, "Cdagup", b + 1, "Cup", b
  os .+= -t1, "Cdagdn", b, "Cdn", b + 1
  os .+= -t1, "Cdagdn", b + 1, "Cdn", b
  os .+= V1, "Ntot", b, "Ntot", b + 1
end
for b in 1:(N - 2)
  os .+= -t2, "Cdagup", b, "Cup", b + 2
  os .+= -t2, "Cdagup", b + 2, "Cup", b
  os .+= -t2, "Cdagdn", b, "Cdn", b + 2
  os .+= -t2, "Cdagdn", b + 2, "Cdn", b
end
H = MPO(os, sites)

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
