using ITensors

include(joinpath("..", "src", "electronk.jl"))
include(joinpath("..", "src", "hubbard.jl"))

function main(; Nx = 6,
                Ny = 3,
                U = 4.0,
                t = 1.0,
                conserve_ky = true)
  N = Nx * Ny

  sweeps = Sweeps(10)
  maxdim!(sweeps, 100, 200, 400, 800, 2000, 3000)
  cutoff!(sweeps, 1e-6)
  noise!(sweeps, 1e-6, 1e-7, 1e-8, 0.0)
  @show sweeps

  sites = siteinds("ElecK", N;
                   conserve_qns = true,
                   conserve_ky = conserve_ky,
                   modulus_ky = Ny)

  ampo = hubbard(Nx = Nx, Ny = Ny, t = t, U = U, ky = true) 
  H = MPO(ampo, sites)

  # Create start state
  state = Vector{String}(undef, N)
  for i in 1:N
    x = (i - 1) ÷ Ny
    y = (i - 1) % Ny
    if x % 2 == 0
      if y % 2 == 0
        state[i] = "Up"
      else
        state[i] = "Dn"
      end
    else
      if y % 2 == 0
        state[i] = "Dn"
      else
        state[i] = "Up"
      end
    end
  end

  psi0 = randomMPS(sites, state, 10)

  energy, psi = dmrg(H, psi0, sweeps)
  @show Nx, Ny
  @show t, U
  @show flux(psi)
  @show maxlinkdim(psi)
  @show energy
end

main()

