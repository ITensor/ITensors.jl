#include "itensor/all.h"
using namespace itensor;

int 
main()
    {
    int N = 100;

    auto sites = SpinOne(N,{"ConserveQNs=",false});

    auto ampo = AutoMPO(sites);
    for(auto j : range1(N-1))
        {
        ampo += 0.5,"S+",j,"S-",j+1;
        ampo += 0.5,"S-",j,"S+",j+1;
        ampo +=     "Sz",j,"Sz",j+1;
        }
    auto H = toMPO(ampo);

    auto state = InitState(sites);
    for(auto i : range1(N))
        {
        if(i%2 == 1) state.set(i,"Up");
        else         state.set(i,"Dn");
        }
    auto psi0 = MPS(state);

    auto sweeps = Sweeps(5);
    sweeps.maxdim() = 10,20,100,100,200;
    sweeps.cutoff() = 1E-11;
    sweeps.niter() = 2;
    sweeps.noise() = 1E-7,1E-8,0.0;
    println(sweeps);

    auto [energy,psi] = dmrg(H,psi0,sweeps,"Quiet");

    PrintData(inner(psi,H,psi));

    return 0;
    }
