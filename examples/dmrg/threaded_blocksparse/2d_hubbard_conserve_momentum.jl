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
  use_splitblocks=true,
  nsweeps=10,
  blas_num_threads=1,
  strided_num_threads=1,
  use_threaded_blocksparse=true,
  outputlevel=1,
  seed=1234,
)
  Random.seed!(seed)
  ITensors.Strided.set_num_threads(strided_num_threads)
  BLAS.set_num_threads(blas_num_threads)
  if use_threaded_blocksparse
    ITensors.enable_threaded_blocksparse()
  else
    ITensors.disable_threaded_blocksparse()
  end

  if outputlevel > 0
    @show Threads.nthreads()
    @show Sys.CPU_THREADS
    @show ITensors.blas_get_num_threads()
    @show ITensors.Strided.get_num_threads()
    @show ITensors.using_threaded_blocksparse()
    println()
  end

  N = Nx * Ny

  maxdims = min.([100, 200, 400, 800, 2000, 3000, maxdim], maxdim)
  cutoff = [1E-6]
  noise = [1E-6, 1E-7, 1E-8, 0.0]

  sites = siteinds("ElecK", N; conserve_qns=true, conserve_ky=conserve_ky, modulus_ky=Ny)

  ampo = hubbard(; Nx=Nx, Ny=Ny, t=t, U=U, ky=true)
  H = MPO(ampo, sites)

  if outputlevel > 0
    @show use_splitblocks
  end

  # This step makes the MPO more sparse but also
  # introduces more blocks.
  # It generally improves DMRG performance
  # at large bond dimensions.
  if use_splitblocks
    H = splitblocks(linkinds, H)
  end

  # Number of structural nonzero elements in a bulk
  # Hamiltonian MPO tensor
  if outputlevel > 0
    @show nnz(H[end ÷ 2])
    @show nnzblocks(H[end ÷ 2])
  end

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

  energy, psi = @time dmrg(
    H, psi0; nsweeps, maxdims, cutoff, noise, outputlevel=outputlevel
  )

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
main(; nsweeps=2, use_threaded_blocksparse=false, outputlevel=0)
println()
println("With threaded block sparse:\n")
main(; nsweeps=2, use_threaded_blocksparse=true, outputlevel=0)
println()

println("################################")
println("Runtime")
println("################################")
println()
println("Without threaded block sparse:\n")
main(; nsweeps=10, use_threaded_blocksparse=false)
println()
println("With threaded block sparse:\n")
main(; nsweeps=10, use_threaded_blocksparse=true)
println()
