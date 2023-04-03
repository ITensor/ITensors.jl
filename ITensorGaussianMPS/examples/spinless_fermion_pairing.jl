# This script shows a minimal example of the GMPS-MPS conversion
# of the ground state of quadratic fermionic Hamiltonian with pairing terms.
using LinearAlgebra
using ITensors
using ITensorGaussianMPS

ITensors.disable_contraction_sequence_optimization()
let
  N = 8
  sites = siteinds("Fermion", N; conserve_qns=false, conserve_nfparity=true)
  _maxlinkdim = 100
  # DMRG cutoff
  _cutoff = 1e-13
  # Hopping
  t = -1.0
  # Electron-electron on-site interaction
  U = 0.0
  # Pairing
  Delta = 1.00
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
  h = quadratic_hamiltonian(os)
  hb = ITensorGaussianMPS.reverse_interleave(h)
  # Make MPO from free fermion Hamiltonian in blocked format
  os_new = OpSum()
  for i in 1:N
    for j in 1:N
      if abs(hb[i, j]) > 1e-8
        os_new .+= -t, "Cdag", i, "C", j
        os_new .+= t, "C", i, "Cdag", j
        os_new .+= Delta / 2.0 * sign(i - j), "C", i, "C", j
        os_new .+= -Delta / 2.0 * sign(i - j), "Cdag", i, "Cdag", j
      end
    end
  end
  H = ITensors.MPO(os_h + os_p, sites)

  #Get Ground state 
  @assert ishermitian(h)
  e = eigvals(Hermitian(h))
  @show e
  E, V = eigen_gaussian(h)
  @show sum(E[1:N])
  Φ = V[:, 1:N]
  c = real.(conj(Φ) * transpose(Φ))

  #Get (G)MPS
  psi = ITensorGaussianMPS.correlation_matrix_to_mps(
    sites, c; eigval_cutoff=1e-10, maxblocksize=14, cutoff=1e-11
  )
  @show eltype(psi[1])
  cdagc = correlation_matrix(psi, "C", "Cdag")
  cc = correlation_matrix(psi, "C", "C")

  println("\nFree fermion starting energy")
  @show flux(psi)
  @show inner(psi, H, psi)
  println("\nRun dmrg with GMPS starting state")
  sweeps = Sweeps(12)
  setmaxdim!(sweeps, 10, 20, 40, _maxlinkdim)
  setcutoff!(sweeps, _cutoff)
  _, psidmrg = dmrg(H, psi, sweeps)
  cdagc_dmrg = correlation_matrix(psidmrg, "C", "Cdag")
  cc_dmrg = correlation_matrix(psidmrg, "C", "C")

  @show norm(cdagc_dmrg - cdagc)
  @show norm(cc_dmrg - cc)

  @show inner(psidmrg, H, psidmrg)
  @show(abs(inner(psidmrg, psi)))

  #return
end
nothing
