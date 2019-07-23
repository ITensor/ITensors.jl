#include "itensor/all.h"
using namespace itensor;

int 
main()
    {
    int Ny = 6;
    int Nx = 12;
    bool yperiodic = false;

    auto N = Nx*Ny;

    auto sites = SpinHalf(N,{"ConserveQNs=",false});

    auto lattice = squareLattice(Nx,Ny,{"YPeriodic=",yperiodic});
    //auto lattice = triangularLattice(Nx,Ny,{"YPeriodic=",yperiodic});

    auto ampo = AutoMPO(sites);
    for(auto b : lattice)
        {
        ampo += 0.5,"S+",b.s1,"S-",b.s2;
        ampo += 0.5,"S-",b.s1,"S+",b.s2;
        ampo +=     "Sz",b.s1,"Sz",b.s2;
        }
    auto H = toMPO(ampo);

    auto state = InitState(sites);
    for(auto i : range1(N))
        {
        if(i%2 == 1) state.set(i,"Up");
        else         state.set(i,"Dn");
        }
    auto psi0 = MPS(state);

    auto sweeps = Sweeps(10);
    sweeps.maxdim() = 10,20,100,100,200,400,800;
    sweeps.cutoff() = 1E-8;
    sweeps.niter() = 2;
    sweeps.noise() = 1E-7,1E-8,0.0;
    println(sweeps);

    auto [energy,psi] = dmrg(H,psi0,sweeps,"Quiet");

    PrintData(inner(psi,H,psi));

    return 0;
    }
