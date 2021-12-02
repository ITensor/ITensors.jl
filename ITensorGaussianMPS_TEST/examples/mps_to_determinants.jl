using ITensors
using ITensorGaussianMPS
using LinearAlgebra

# Half filling
N = 20
Nf_up = N ÷ 2
Nf_dn = N ÷ 2
Nf = Nf_up + Nf_dn

@show N, Nf

# Maximum MPS link dimension
_maxlinkdim = 50

@show _maxlinkdim

# DMRG cutoff
_cutoff = 1e-8

# Hopping
t = 1.0

# Electron-electron on-site interaction
U = 1.0

@show t, U

# Make the free fermion Hamiltonian for the up spins
os_up = OpSum()
for n in 1:(N - 1)
  os_up .+= -t, "Cdagup", n, "Cup", n + 1
  os_up .+= -t, "Cdagup", n + 1, "Cup", n
end

# Make the free fermion Hamiltonian for the down spins
os_dn = OpSum()
for n in 1:(N - 1)
  os_dn .+= -t, "Cdagdn", n, "Cdn", n + 1
  os_dn .+= -t, "Cdagdn", n + 1, "Cdn", n
end

# Hopping Hamiltonians for the up and down spins
h_up = hopping_hamiltonian(os_up)
h_dn = hopping_hamiltonian(os_dn)

# Get the Slater determinant
Φ_up = slater_determinant_matrix(h_up, Nf_up)
Φ_dn = slater_determinant_matrix(h_dn, Nf_dn)

# Create an MPS from the slater determinants.
s = siteinds("Electron", N; conserve_qns=true)
println("Making free fermion starting MPS")
@time ψ0 = slater_determinant_to_mps(
  s, Φ_up, Φ_dn; eigval_cutoff=1e-4, cutoff=_cutoff, maxdim=_maxlinkdim
)
@show maxlinkdim(ψ0)

# The total interacting Hamiltonian
os = os_up + os_dn
for n in 1:N
  os .+= U, "Nupdn", n
end
H = MPO(os, s)

println("Free fermion starting state energy")
@show flux(ψ0)
@show inner(ψ0, H, ψ0)

println("\nStart from free fermion state")
sweeps = Sweeps(5)
setmaxdim!(sweeps, _maxlinkdim)
setcutoff!(sweeps, _cutoff)
e, ψ = @time dmrg(H, ψ0, sweeps)
@show e
@show flux(ψ)

using ITensorGaussianMPS: correlation_matrix_to_gmps, correlation_matrix_to_mps, entropy

Λ_up = correlation_matrix(ψ, "Cdagup", "Cup")
Λ_dn = correlation_matrix(ψ, "Cdagdn", "Cdn")
ψ̃0 = correlation_matrix_to_mps(s, Λ_up, Λ_dn; eigval_cutoff=1e-2, maxblocksize=4)
@show inner(ψ̃0, ψ)
@show inner(ψ̃0, H, ψ̃0)
