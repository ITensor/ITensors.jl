using Compat
using ITensors
using Test
using LinearAlgebra

if isone(Threads.nthreads())
  @warn "Testing block sparse multithreading but only one thread is set!"
end

@testset "Threading" begin
  blas_num_threads = Compat.get_num_threads()
  strided_num_threads = ITensors.NDTensors.Strided.get_num_threads()

  BLAS.set_num_threads(1)
  ITensors.NDTensors.Strided.set_num_threads(1)

  @testset "Bug fixed in threaded block sparse" begin
    maxdim = 10
    nsweeps = 2
    outputlevel = 0
    cutoff = 0.0
    Nx = 4
    Ny = 2
    U = 4.0
    t = 1.0
    N = Nx * Ny
    sweeps = Sweeps(nsweeps)
    maxdims = min.(maxdim, [100, 200, 400, 800, 2000, 3000, maxdim])
    maxdim!(sweeps, maxdims...)
    cutoff!(sweeps, cutoff)
    noise!(sweeps, 1e-6, 1e-7, 1e-8, 0.0)
    sites = siteinds("Electron", N; conserve_qns=true)
    lattice = square_lattice(Nx, Ny; yperiodic=true)
    ampo = OpSum()
    for b in lattice
      ampo .+= -t, "Cdagup", b.s1, "Cup", b.s2
      ampo .+= -t, "Cdagup", b.s2, "Cup", b.s1
      ampo .+= -t, "Cdagdn", b.s1, "Cdn", b.s2
      ampo .+= -t, "Cdagdn", b.s2, "Cdn", b.s1
    end
    for n in 1:N
      ampo .+= U, "Nupdn", n
    end
    H = MPO(ampo, sites)
    Hsplit = splitblocks(linkinds, H)
    state = [isodd(n) ? "↑" : "↓" for n in 1:N]
    ψ0 = productMPS(sites, state)
    enabled = ITensors.enable_threaded_blocksparse(true)
    energy, _ = dmrg(H, ψ0, sweeps; outputlevel=outputlevel)
    energy_split, _ = dmrg(Hsplit, ψ0, sweeps; outputlevel=outputlevel)
    @test energy_split ≈ energy
    ITensors.enable_threaded_blocksparse(enabled)
  end

  BLAS.set_num_threads(blas_num_threads)
  ITensors.NDTensors.Strided.set_num_threads(strided_num_threads)
end
