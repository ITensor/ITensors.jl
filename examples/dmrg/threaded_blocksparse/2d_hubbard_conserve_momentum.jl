using ITensors
using LinearAlgebra
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
  nsweeps=10,
  blas_num_threads=1,
  strided_num_threads=1,
  threaded_blocksparse=false,
  outputlevel=1,
  seed=1234,
)
  Random.seed!(seed)
  ITensors.Strided.set_num_threads(strided_num_threads)
  BLAS.set_num_threads(blas_num_threads)
  ITensors.enable_threaded_blocksparse(threaded_blocksparse)

  if outputlevel > 0
    @show Threads.nthreads()
    @show Sys.CPU_THREADS
    @show BLAS.get_num_threads()
    @show ITensors.Strided.get_num_threads()
    @show ITensors.using_threaded_blocksparse()
    println()
  end

  N = Nx * Ny

  maxdim = min.([100, 200, 400, 800, 2000, 3000, maxdim], maxdim)
  cutoff = [1E-6]
  noise = [1E-6, 1E-7, 1E-8, 0.0]

  sites = siteinds("ElecK", N; conserve_qns=true, conserve_ky, modulus_ky=Ny)

  os = hubbard(; Nx, Ny, t, U, ky=true)
  H = MPO(os, sites)

  # Number of structural nonzero elements in a bulk
  # Hamiltonian MPO tensor
  if outputlevel > 0
    @show nnz(H[end ÷ 2])
    @show nnzblocks(H[end ÷ 2])
  end

  # Create starting state with checkerboard
  # pattern
  state = map(CartesianIndices((Ny, Nx))) do I
    return iseven(I[1]) ⊻ iseven(I[2]) ? "↓" : "↑"
  end
  display(state)

  psi0 = randomMPS(sites, state; linkdim=10)

  energy, psi = @time dmrg(H, psi0; nsweeps, maxdim, cutoff, noise, outputlevel)

  if outputlevel > 0
    @show Nx, Ny
    @show t, U
    @show flux(psi)
    @show maxlinkdim(psi)
    @show energy
  end
  return nothing
end

println("################################")
println("Compilation")
println("################################")
println("Without threaded block sparse:\n")
main(; nsweeps=2, threaded_blocksparse=false, outputlevel=0)
println()
println("With threaded block sparse:\n")
main(; nsweeps=2, threaded_blocksparse=true, outputlevel=0)
println()

println("################################")
println("Runtime")
println("################################")
println()
println("Without threaded block sparse:\n")
main(; nsweeps=10, threaded_blocksparse=false)
println()
println("With threaded block sparse:\n")
main(; nsweeps=10, threaded_blocksparse=true)
println()
