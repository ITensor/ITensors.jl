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
        add!(ampo,"Sz",j,"Sz",j+1)
        add!(ampo,h,"Sx",j)
    end
    add!(ampo,h,"Sx",length(sites))
    # Convert these terms to an MPO tensor network
    return toMPO(ampo)
end

N = 20
sites = spinHalfSites(N)
psi0 = randomMPS(sites)
sweeps = Sweeps(5)
maxdim!(sweeps, 10,20,100,100,200)
cutoff!(sweeps, 1E-10)

#=
create observer which will measure Sᶻ at each
site during the dmrg sweeps and track energies after each sweep.
in addition it will stop the computation if energy converges within
1e-8 tolerance
=#
observer= DMRGObserver(["Sz"],sites,1e-4)

# Run the DMRG algorithm, returning energy and optimized MPS
energy, psi = dmrg(tfimMPO(0.1,sites),psi0, sweeps,observer=observer)
# @printf("Final energy = %.12f\n",energy)

for Szs in measurements(observer)["Sz"]
    println(Szs)
end
