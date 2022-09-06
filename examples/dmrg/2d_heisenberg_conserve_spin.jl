using ITensors

let
  Ny = 6
  Nx = 12

  N = Nx * Ny

  sites = siteinds("S=1/2", N; conserve_qns=true)

  lattice = square_lattice(Nx, Ny; yperiodic=false)

  os = OpSum()
  for b in lattice
    os .+= 0.5, "S+", b.s1, "S-", b.s2
    os .+= 0.5, "S-", b.s1, "S+", b.s2
    os .+= "Sz", b.s1, "Sz", b.s2
  end
  H = MPO(os, sites)

  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  # Initialize wavefunction to a random MPS
  # of bond-dimension 10 with same quantum
  # numbers as `state`
  psi0 = randomMPS(sites, state, 20)

  nsweeps = 10
  maxdim = [20, 60, 100, 100, 200, 400, 800]
  cutoff = [1E-8]

  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)

  return nothing
end
