using Compat
using ITensors
using Test
using LinearAlgebra

@testset "Threading" begin
  blas_num_threads = Compat.get_num_threads()
  strided_num_threads = ITensors.NDTensors.Strided.get_num_threads()

  BLAS.set_num_threads(1)
  ITensors.NDTensors.Strided.set_num_threads(1)

  @testset "Threaded contraction" begin
    i = Index([QN(0) => 500, QN(1) => 500])
    A = randomITensor(i', dag(i))

    using_threaded_blocksparse = ITensors.disable_threaded_blocksparse()

    R = A' * A

    ITensors.enable_threaded_blocksparse()

    Rthreaded = A' * A

    @test R ≈ Rthreaded

    #ITensors.disable_threaded_blocksparse()
    #time = @elapsed B = A' * A
    #ITensors.enable_threaded_blocksparse()
    #time_threaded = @elapsed B = A' * A
    #@test time > time_threaded

    if !using_threaded_blocksparse
      ITensors.disable_threaded_blocksparse()
    end
  end

  @testset "Contraction resulting in no blocks with threading bug" begin
    i = Index([QN(0) => 1, QN(1) => 1])
    A = emptyITensor(i', dag(i))
    B = emptyITensor(i', dag(i))
    A[i' => 1, i => 1] = 11.0
    B[i' => 2, i => 2] = 22.0

    using_threaded_blocksparse = ITensors.disable_threaded_blocksparse()
    C1 = A' * B
    ITensors.enable_threaded_blocksparse()
    C2 = A' * B
    if !using_threaded_blocksparse
      ITensors.disable_threaded_blocksparse()
    end

    @test nnzblocks(C1) == 0
    @test nnzblocks(C2) == 0
    @test nnz(C1) == 0
    @test nnz(C2) == 0
    @test C1 ≈ C2
  end

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
    using_threaded_blocksparse = ITensors.enable_threaded_blocksparse()
    energy, _ = dmrg(H, ψ0, sweeps; outputlevel=outputlevel)
    energy_split, _ = dmrg(Hsplit, ψ0, sweeps; outputlevel=outputlevel)
    @test energy_split ≈ energy
    if !using_threaded_blocksparse
      ITensors.disable_threaded_blocksparse()
    end
  end

  BLAS.set_num_threads(blas_num_threads)
  ITensors.NDTensors.Strided.set_num_threads(strided_num_threads)
end
