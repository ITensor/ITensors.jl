using ITensors, ITensors.CuITensors

function main()
    Ny = 6
    Nx = 6

    N = Nx*Ny

    sites = spinHalfSites(N;conserveQNs=false)

    lattice = squareLattice(Nx,Ny,yperiodic=false)

    ampo = AutoMPO(sites)
    for b in lattice
        add!(ampo,0.5,"S+",b.s1,"S-",b.s2)
        add!(ampo,0.5,"S-",b.s1,"S+",b.s2)
        add!(ampo,    "Sz",b.s1,"Sz",b.s2)
    end
    H = toMPO(ampo)

    state = InitState(sites)
    for n=1:N
    state[n] = isodd(n) ? "Up" : "Dn"
    end
    psi0 = MPS(state)
    cH = cuMPO(H)
    cpsi0 = cuMPS(psi0)

    sweeps = Sweeps(5)
    maxdim!(sweeps,10,20,100,200,400)
    cutoff!(sweeps,1E-8)
    @show sweeps
    cenergy,cpsi = dmrg(cH,cpsi0,sweeps)
end
main()
