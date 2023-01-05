"""
This script shows a minimal example of the GMPS-MPS conversion
of the ground state of quadratic fermionic Hamiltonian with pairing terms.
It also provides an example for an instability of the GMPS-MPS conversion
for certain correlation matrices (presumably measure 0),
which is resolved by an infinitesimal perturbation.
"""

using LinearAlgebra
using ITensors
using ITensorGaussianMPS
ITensors.disable_contraction_sequence_optimization()
let
  # Half filling
  N = 10
  Nf = N ÷ 2
  #Nf= 
  @show N
  sites = siteinds("Fermion", N; conserve_qns=false)
  _maxlinkdim = 100
  # DMRG cutoff
  _cutoff = 1e-13
  # Hopping
  t = -1.2
  # Electron-electron on-site interaction
  U = 0.0
  # Pairing
  Delta = 0.7
  @show t, U, Delta
  # Free fermion Hamiltonian
  os_h = OpSum()
  for n in 1:(N - 1)
    os_h .+= -t, "Cdag", n, "C", n + 1
    os_h .+= -t, "Cdag", n + 1, "C", n
  end
  os_p = OpSum()
  for n in 1:(N - 1)
    os_p .+= Delta / 2.0, "Cdag", n, "Cdag", n + 1
    os_p .+= -Delta / 2.0, "Cdag", n + 1, "Cdag", n
    os_p .+= -Delta / 2.0, "C", n, "C", n + 1
    os_p .+= Delta / 2.0, "C", n + 1, "C", n
  end
  os = os_h + os_p
  h, hb = ITensorGaussianMPS.pairing_hamiltonian(os_h, os_p)

  # Make MPO from free fermion Hamiltonian
  os_new = OpSum()
  for i in 1:N
    for j in 1:N
      if abs(hb[i, j]) > 1e-8
        os_new .+= hb[N + i, N + j], "Cdag", i, "C", j
        os_new .+= hb[i, j], "C", i, "Cdag", j
        os_new .+= hb[i, N + j], "C", i, "C", j
        os_new .+= hb[N + i, j], "Cdag", i, "Cdag", j
      end
    end
  end
  H = ITensors.MPO(os_new, sites)

  #Get Ground state 
  E, V = eigen(Hermitian(h))
  Φ = V[:, 1:N]
  c = conj(Φ) * transpose(Φ)

  #Get (G)MPS
  Gamma = ITensorGaussianMPS.reverse_interleave(c)
  pert = 1e-14 * rand(2 * N, 2 * N)
  sympert = 0.5 * (pert + transpose(pert))
  psi = ITensorGaussianMPS.correlation_matrix_to_mps(
    sites,
    ITensorGaussianMPS.interleave(Complex.(Gamma));
    eigval_cutof=1e-10,
    maxblocksize=14,
    cutoff=1e-11,
  )
  psi_perturbed = ITensorGaussianMPS.correlation_matrix_to_mps(
    sites,
    ITensorGaussianMPS.interleave(Complex.(Gamma)) + sympert;
    eigval_cutof=1e-10,
    maxblocksize=14,
    cutoff=1e-11,
  )
  @show eltype(psi[1])
  cdagc = correlation_matrix(psi, "C", "Cdag")
  println("\nFree fermion starting energy")
  @show flux(psi)
  @show inner(psi, H, psi)
  @show inner(psi_perturbed, H, psi_perturbed)

  println("\nRun dmrg with GMPS starting state")
  sweeps = Sweeps(12)
  setmaxdim!(sweeps, 10, 20, 40, _maxlinkdim)
  setcutoff!(sweeps, _cutoff)
  _, psidmrg = dmrg(H, copy(psi_perturbed), sweeps)
  @show inner(psidmrg, H, psidmrg)
  @show(abs(inner(psidmrg, psi)))
  @show(abs(inner(psidmrg, psi_perturbed)))

  #return
end
nothing
