using ITensors
using Random

include(joinpath(ITensors.examples_dir(), "src", "electronk.jl"))
include(joinpath(ITensors.examples_dir(), "src", "hubbard.jl"))

function main(;
  Nx::Int=6,
  Ny::Int=3,
  U::Float64=4.0,
  t::Float64=1.0,
  maxdim::Int=3000,
  conserve_ky=true,
  use_splitblocks=true,
  seed=1234,
)
  Random.seed!(seed)
  @show Threads.nthreads()
  @show ITensors.using_threaded_blocksparse()

  N = Nx * Ny

  nsweeps = 10
  maxdims = min.([100, 200, 400, 800, 2000, 3000, maxdim], maxdim)
  cutoff = [1E-6]
  noise = [1E-6, 1E-7, 1E-8, 0.0]

  sites = siteinds("ElecK", N; conserve_qns=true, conserve_ky=conserve_ky, modulus_ky=Ny)

  ampo = hubbard(; Nx=Nx, Ny=Ny, t=t, U=U, ky=true)
  H = MPO(ampo, sites)

  # This step makes the MPO more sparse.
  # It generally improves DMRG performance
  # at large bond dimensions but makes DMRG slower at
  # small bond dimensions.
  if use_splitblocks
    H = splitblocks(linkinds, H)
  end

  # Number of structural nonzero elements in a bulk
  # Hamiltonian MPO tensor
  @show nnz(H[end ÷ 2])
  @show nnzblocks(H[end ÷ 2])

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

  energy, psi = @time dmrg(H, psi0; nsweeps, maxdims, cutoff, noise, svd_alg="divide_and_conquer")
  @show Nx, Ny
  @show t, U
  @show flux(psi)
  @show maxlinkdim(psi)
  @show energy
  return energy, H, psi
end
