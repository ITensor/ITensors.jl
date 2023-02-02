using ITensors
using LinearAlgebra
using Random

include(joinpath(ITensors.examples_dir(), "src", "electronk.jl"))
include(joinpath(ITensors.examples_dir(), "src", "hubbard.jl"))

"""
Usage:
```julia
main(; Nx=8, Ny=4, U=4.0, t=1.0, nsweeps=10, maxdim=3000, threaded_blocksparse=false);
main(; Nx=8, Ny=4, U=4.0, t=1.0, nsweeps=10, maxdim=3000, threaded_blocksparse=true);
```
"""
function main(;
  Nx::Int=8,
  Ny::Int=4,
  U::Float64=4.0,
  t::Float64=1.0,
  maxdim::Int=3000,
  conserve_ky=true,
  threaded_blocksparse=false,
  nsweeps=10,
  seed=1234,
)
  # Helps make results reproducible when comparing
  # sequential vs. threaded.
  itensor_rng = Xoshiro()
  Random.seed!(itensor_rng, seed)

  @show Threads.nthreads()

  # Disable other threading
  BLAS.set_num_threads(1)
  ITensors.Strided.set_num_threads(1)

  ITensors.enable_threaded_blocksparse(threaded_blocksparse)
  @show ITensors.using_threaded_blocksparse()

  N = Nx * Ny

  maxdim = min.([100, 200, 400, 800, 2000, 3000, maxdim], maxdim)
  cutoff = [1e-6]
  noise = [1e-6, 1e-7, 1e-8, 0.0]

  sites = siteinds("ElecK", N; conserve_qns=true, conserve_ky, modulus_ky=Ny)

  os = hubbard(; Nx, Ny, t, U, ky=true)
  H = MPO(os, sites)

  # Number of structural nonzero elements in a bulk
  # Hamiltonian MPO tensor
  @show nnz(H[end ÷ 2])
  @show nnzblocks(H[end ÷ 2])

  # Create starting state with checkerboard
  # pattern
  state = map(CartesianIndices((Ny, Nx))) do I
    return iseven(I[1]) ⊻ iseven(I[2]) ? "↓" : "↑"
  end
  display(state)

  psi0 = randomMPS(itensor_rng, sites, state; linkdims=2)
  @time @show inner(psi0', H, psi0)

  energy, psi = @time dmrg(H, psi0; nsweeps, maxdim, cutoff, noise)
  @show Nx, Ny
  @show t, U
  @show flux(psi)
  @show maxlinkdim(psi)
  @show energy
  return energy, H, psi
end
