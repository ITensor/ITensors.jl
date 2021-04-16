using ITensors

function main(; Nx = 6,
                Ny = 3,
                U = 4.0,
                t = 1.0)
  N = Nx * Ny

  sweeps = Sweeps(10)
  setmaxdim!(sweeps, 100, 200, 400, 800, 1600)
  setcutoff!(sweeps, 1e-6)
  setnoise!(sweeps, 1e-6, 1e-7, 1e-8, 0.0)
  @show sweeps

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

  energy,psi = dmrg(H, psi0, sweeps)
  @show t, U
  @show flux(psi)
  @show maxlinkdim(psi)
  @show energy

  return
end

main()

