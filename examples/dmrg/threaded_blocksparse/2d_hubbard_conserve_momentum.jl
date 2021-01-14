using ITensors
using LinearAlgebra
using Random

include(joinpath(ITensors.examples_dir(), "src", "electronk.jl"))
include(joinpath(ITensors.examples_dir(), "src", "hubbard.jl"))

function main(; Nx::Int = 6,
                Ny::Int = 3,
                U::Float64 = 4.0,
                t::Float64 = 1.0,
                maxdim::Int = 3000,
                conserve_ky = true,
                use_splitblocks = true,
                nsweeps = 10,
                blas_num_threads = 1, #Sys.CPU_THREADS,
                use_threaded_blocksparse = true,
                outputlevel = 1)
  Random.seed!(1234)
  ITensors.Strided.set_num_threads(1)
  BLAS.set_num_threads(blas_num_threads)
  if use_threaded_blocksparse
    ITensors.enable_threaded_blocksparse()
  else
    ITensors.disable_threaded_blocksparse()
  end

  @show Threads.nthreads()
  @show blas_num_threads
  @show ITensors.Strided.get_num_threads()
  @show ITensors.using_threaded_blocksparse()
  println()

  N = Nx * Ny

  sweeps = Sweeps(nsweeps)
  maxdims = min.([100, 200, 400, 800, 2000, 3000, maxdim], maxdim)
  maxdim!(sweeps, maxdims...)
  cutoff!(sweeps, 1e-6)
  noise!(sweeps, 1e-6, 1e-7, 1e-8, 0.0)
  @show sweeps

  sites = siteinds("ElecK", N;
                   conserve_qns = true,
                   conserve_ky = conserve_ky,
                   modulus_ky = Ny)

  ampo = hubbard(Nx = Nx, Ny = Ny, t = t, U = U, ky = true) 
  H = MPO(ampo, sites)

  @show use_splitblocks

  # This step makes the MPO more sparse.
  # It generally improves DMRG performance
  # at large bond dimensions but makes DMRG slower at
  # small bond dimensions.
  if use_splitblocks
    H = splitblocks(linkinds, H)
  end

  # Number of structural nonzero elements in a bulk
  # Hamiltonian MPO tensor
  @show nnz(H[end÷2])
  @show nnzblocks(H[end÷2])

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

  energy, psi = @time dmrg(H, psi0, sweeps; outputlevel = outputlevel)
  @show Nx, Ny
  @show t, U
  @show flux(psi)
  @show maxlinkdim(psi)
  @show energy
end

println("################################")
println("Compilation")
println("################################")
println("Without threaded block sparse:\n")
main(nsweeps = 2, use_splitblocks = true, use_threaded_blocksparse = false, outputlevel = 0)
println()
println("With threaded block sparse:\n")
main(nsweeps = 2, use_splitblocks = true, use_threaded_blocksparse = true, outputlevel = 0)
println()

println("################################")
println("Runtime")
println("################################")
println()
println("Without threaded block sparse:\n")
main(nsweeps = 10, use_splitblocks = false, use_threaded_blocksparse = false)
println()
println("With threaded block sparse:\n")
main(nsweeps = 10, use_splitblocks = false, use_threaded_blocksparse = true)
println()


