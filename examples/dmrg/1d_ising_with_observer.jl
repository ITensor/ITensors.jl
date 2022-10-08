# In this example we show how to pass a DMRGObserver to
# the dmrg function which allows tracking energy convergence and
# convergence of local operators.
using ITensors

"""
  Get MPO of transverse field Ising model Hamiltonian with field strength h
"""
function tfimMPO(sites, h::Float64)
  # Input operator terms which define a Hamiltonian
  N = length(sites)
  os = OpSum()
  for j in 1:(N - 1)
    os += -1, "Z", j, "Z", j + 1
  end
  for j in 1:N
    os += h, "X", j
  end
  # Convert these terms to an MPO tensor network
  return MPO(os, sites)
end

let
  N = 100
  sites = siteinds("S=1/2", N)
  psi0 = randomMPS(sites; linkdims=10)

  # define parameters for DMRG sweeps
  nsweeps = 15
  maxdim = [10, 20, 100, 100, 200]
  cutoff = [1E-10]

  #=
  create observer which will measure Sᶻ at each
  site during the dmrg sweeps and track energies after each sweep.
  in addition it will stop the computation if energy converges within
  1E-7 tolerance
  =#
  let
    Sz_observer = DMRGObserver(["Sz"], sites; energy_tol=1E-7)

    # we will now run DMRG calculation for different values
    # of the transverse field and check how local observables
    # converge to their ground state values

    println("Running DMRG for TFIM with h=0.1")
    println("================================")
    H = tfimMPO(sites, 0.1)
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, observer=Sz_observer)

    for (i, Szs) in enumerate(measurements(Sz_observer)["Sz"])
      println("<Σ Sz> after sweep $i = ", sum(Szs) / N)
    end
  end

  let
    println("\nRunning DMRG for TFIM with h=1.0 (critical point)")
    println("================================")
    Sz_observer = DMRGObserver(["Sz"], sites; energy_tol=1E-7)
    H = tfimMPO(sites, 1.0)
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, observer=Sz_observer)

    for (i, Szs) in enumerate(measurements(Sz_observer)["Sz"])
      println("<Σ Sz> after sweep $i = ", sum(Szs) / N)
    end
  end

  let
    println("\nRunning DMRG for TFIM with h=5.")
    println("================================")
    Sz_Sx_observer = DMRGObserver(["Sz", "Sx"], sites; energy_tol=1E-7)
    H = tfimMPO(sites, 5.0)
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, observer=Sz_Sx_observer)

    for (i, Szs) in enumerate(measurements(Sz_Sx_observer)["Sz"])
      println("<Σ Sz> after sweep $i = ", sum(Szs) / N)
    end
    println()
    for (i, Sxs) in enumerate(measurements(Sz_Sx_observer)["Sx"])
      println("<Σ Sx> after sweep $i = ", sum(Sxs) / N)
    end
  end
end
