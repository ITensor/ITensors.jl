# In this example we show how to pass a DMRGObserver to
# the dmrg function which allows tracking energy convergence and
# convergence of local operators.
using ITensors


"""
get MPO of transverse field Ising model Hamiltonian with field strength h
"""
function tfimMPO(h,sites)
    # Input operator terms which define a Hamiltonian
    ampo = AutoMPO(sites)
    for j=1:length(sites)-1
        add!(ampo,-1.,"Sz",j,"Sz",j+1)
        add!(ampo,h,"Sx",j)
    end
    add!(ampo,h,"Sx",length(sites))
    # Convert these terms to an MPO tensor network
    return toMPO(ampo)
end

N = 100
sites = spinHalfSites(N)
psi0 = randomMPS(sites)

# define parameters for DMRG sweeps
sweeps = Sweeps(5)
maxdim!(sweeps, 10,20,100,100,200)
cutoff!(sweeps, 1E-10)

#=
create observer which will measure Sᶻ at each
site during the dmrg sweeps and track energies after each sweep.
in addition it will stop the computation if energy converges within
1e-6 tolerance
=#
observer= DMRGObserver(["Sz"],sites,1e-6)

# we will now run DMRG calculation for different values
# of the transverse field and check how local observables
# converge to their ground state values

println("Running DMRG for TFIM with h=0.1")
println("================================")
energy, psi = dmrg(tfimMPO(0.1,sites),psi0, sweeps,observer=observer)

for (i,Szs) in enumerate(measurements(observer)["Sz"])
    println("magnetization after sweep $i = ", sum(Szs)/N)
end


println("\nRunning DMRG for TFIM with h=1")
println("================================")
observer= DMRGObserver(["Sz"],sites,1e-6)
energy, psi = dmrg(tfimMPO(1.,sites),psi0, sweeps,observer=observer)

for (i,Szs) in enumerate(measurements(observer)["Sz"])
    println("magnetization after sweep $i = ", sum(Szs)/N)
end

println("\nRunning DMRG for TFIM with h=5.")
println("================================")
observer= DMRGObserver(["Sz","Sx"],sites,1e-6)
energy, psi = dmrg(tfimMPO(5.,sites),psi0, sweeps,observer=observer)

for (i,Sxs) in enumerate(measurements(observer)["Sx"])
    println("<Sˣ> after sweep $i = ", sum(Sxs)/N)
end
