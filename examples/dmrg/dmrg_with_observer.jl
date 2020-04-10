# In this example we show how to pass a DMRGObserver to
# the dmrg function which allows tracking energy convergence and
# convergence of local operators.
using ITensors


"""
  Get MPO of transverse field Ising model Hamiltonian with field strength h
"""
function tfimMPO(sites,
                 h::Float64)
  # Input operator terms which define a Hamiltonian
  N = length(sites)
  ampo = AutoMPO()
  for j=1:N-1
    ampo += (-1.,"Sz",j,"Sz",j+1)
    ampo += (h,"Sx",j)
  end
  ampo += (h,"Sx",N)
  # Convert these terms to an MPO tensor network
  return toMPO(ampo,sites)
end

let
  N = 100
  sites = siteinds("S=1/2",N)
  psi0 = randomMPS(sites,10)

  # define parameters for DMRG sweeps
  sweeps = Sweeps(15)
  maxdim!(sweeps, 10,20,100,100,200)
  cutoff!(sweeps, 1E-10)

  #=
  create observer which will measure Sᶻ at each
  site during the dmrg sweeps and track energies after each sweep.
  in addition it will stop the computation if energy converges within
  1E-7 tolerance
  =#
  let
    Sz_observer = DMRGObserver(["Sz"],sites,1E-7)

    # we will now run DMRG calculation for different values
    # of the transverse field and check how local observables
    # converge to their ground state values

    println("Running DMRG for TFIM with h=0.1")
    println("================================")
    H = tfimMPO(sites,0.1)
    energy, psi = dmrg(H,psi0,sweeps,observer=Sz_observer)

    for (i,Szs) in enumerate(measurements(Sz_observer)["Sz"])
      println("<Σ Sz> after sweep $i = ", sum(Szs)/N)
    end
  end


  let
    println("\nRunning DMRG for TFIM with h=0.5 (critical point)")
    println("================================")
    Sz_observer= DMRGObserver(["Sz"],sites,1E-7)
    H = tfimMPO(sites,0.5)
    energy, psi = dmrg(H,psi0,sweeps,observer=Sz_observer)

    for (i,Szs) in enumerate(measurements(Sz_observer)["Sz"])
      println("<Σ Sz> after sweep $i = ", sum(Szs)/N)
    end
  end

  let
    println("\nRunning DMRG for TFIM with h=5.")
    println("================================")
    Sz_Sx_observer= DMRGObserver(["Sz","Sx"],sites,1E-7)
    H = tfimMPO(sites,5.0)
    energy, psi = dmrg(H,psi0,sweeps,observer=Sz_Sx_observer)

    for (i,Szs) in enumerate(measurements(Sz_Sx_observer)["Sz"])
      println("<Σ Sz> after sweep $i = ", sum(Szs)/N)
    end
    println()
    for (i,Sxs) in enumerate(measurements(Sz_Sx_observer)["Sx"])
      println("<Σ Sx> after sweep $i = ", sum(Sxs)/N)
    end
  end
end
