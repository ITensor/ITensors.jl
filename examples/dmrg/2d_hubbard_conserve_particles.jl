using ITensors

function main(; Nx = 4,
                Ny = 4,
                U = 4.0)
  t = 1.0

  N = Nx * Ny

  sites = siteinds("Electron", N;
                   conserve_qns = true)

  lattice = square_lattice(Nx, Ny;
                           yperiodic = true)

  ampo = AutoMPO()
  for b in lattice
    ampo += -t, "Cdagup", b.s1, "Cup", b.s2
    ampo += -t, "Cdagup", b.s2, "Cup", b.s1
    ampo += -t, "Cdagdn", b.s1, "Cdn", b.s2
    ampo += -t, "Cdagdn", b.s2, "Cdn", b.s1
  end
  for n in 1:N
    ampo += U, "Nupdn", n
  end
  H = MPO(ampo,sites)

  # Half filling
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]

  # Initialize wavefunction to a random MPS
  # of bond-dimension 10 with same quantum
  # numbers as `state`
  psi0 = randomMPS(sites, state)

  sweeps = Sweeps(20)
  maxdim!(sweeps, 20, 60, 100, 100, 200, 400, 800)
  cutoff!(sweeps, 1e-8)
  @show sweeps

  energy,psi = dmrg(H, psi0, sweeps)
  @show t, U
  @show flux(psi)
  @show maxlinkdim(psi)
  @show energy

  return
end

main()

