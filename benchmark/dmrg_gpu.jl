using ITensors, ITensors.CuITensors, Printf

function main()
    N = 100
    sites = spinOneSites(N)

    ampo = AutoMPO(sites)
    for j=1:N-1
        add!(ampo,"Sz",j,"Sz",j+1)
        add!(ampo,0.5,"S+",j,"S-",j+1)
        add!(ampo,0.5,"S-",j,"S+",j+1)
    end
    H = toMPO(ampo)
    psi = randomMPS(sites)
    cH = cuMPO(H)
    cpsi = cuMPS(psi)

    sweeps = Sweeps(5)
    maxdim!(sweeps,10,20,100,100,200)
    cutoff!(sweeps,1E-11)
    @show sweeps

    energy,cpsi = @time dmrg(cH,cpsi,sweeps,maxiter=2)

    @printf "Final energy = %.12f\n" energy
    #G.S. energy for N=10,S=1/2 Heisenberg is -4.258035206805
main()
