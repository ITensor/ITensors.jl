using ITensors,
      Printf

include("heisenberg.jl")

let
  N = 50

  s = [Index(QN(1)=>1,QN(-1)=>1; tags="S=1/2,n=$n") for n in 1:N]

  H = heisenberg(s)

  psi0 = MPS(N)

  state = [isodd(n) ? 1 : 2 for n in 1:N] 

  l = [Index(QN(0)=>1; tags="l=$l") for l in 1:N-1]

  psi0[1] = ITensor(s[1],l[1])
  psi0[1][s[1](state[1]),l[1](1)] = 1.0
  for n in 2:N-1
    psi0[n] = ITensor(s[n],dag(l[n-1]),l[n])
    psi0[n][s[n](state[n]),l[n-1](1),l[n](1)] = 1.0
  end
  psi0[N] = ITensor(s[N],dag(l[N-1]))
  psi0[N][s[N](state[N]),l[N-1](1)] = 1.0

  # Plan to do 5 DMRG sweeps:
  sweeps = Sweeps(5)
  # Set maximum MPS bond dimensions for each sweep
  maxdim!(sweeps, 10,20,100,100,200)
  # Set maximum truncation error allowed when adapting bond dimensions
  cutoff!(sweeps, 1E-10)
  @show sweeps

  # Run the DMRG algorithm, returning energy and optimized MPS
  energy, psi = dmrg(H,psi0, sweeps; which_factorization="svd")
  @printf("Final energy = %.12f\n",energy)
end

