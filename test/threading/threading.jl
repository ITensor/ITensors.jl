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

  @testset "Getting and setting global flags" begin
    enabled0 = ITensors.enable_threaded_blocksparse(false)
    @test !ITensors.using_threaded_blocksparse()
    enabled1 = ITensors.enable_threaded_blocksparse(true)
    @test !enabled1
    @test ITensors.using_threaded_blocksparse()
    enabled2 = ITensors.enable_threaded_blocksparse(false)
    @test enabled2
    @test !ITensors.using_threaded_blocksparse()
    enabled3 = ITensors.enable_threaded_blocksparse(enabled0)
    @test !enabled3
    @test ITensors.using_threaded_blocksparse() == enabled0
  end

  @testset "Threaded contraction" begin
    i = Index([QN(0) => 500, QN(1) => 500])
    A = randomITensor(i', dag(i))

    enabled = ITensors.disable_threaded_blocksparse()
    R = A' * A
    ITensors.enable_threaded_blocksparse()
    Rthreaded = A' * A
    @test R ≈ Rthreaded
    if !enabled
      ITensors.disable_threaded_blocksparse()
    end

    # New interface
    enabled = ITensors.enable_threaded_blocksparse(false)
    R = A' * A
    ITensors.enable_threaded_blocksparse(true)
    Rthreaded = A' * A
    @test R ≈ Rthreaded
    ITensors.enable_threaded_blocksparse(enabled)

    # TODO: Test timing?
    # ITensors.enable_threaded_blocksparse(false)
    # time = @elapsed B = A' * A
    # ITensors.enable_threaded_blocksparse(true)
    # time_threaded = @elapsed B = A' * A
    # @test time > time_threaded

  end

  @testset "Contraction resulting in no blocks with threading bug" begin
    i = Index([QN(0) => 1, QN(1) => 1])
    A = emptyITensor(i', dag(i))
    B = emptyITensor(i', dag(i))
    A[i' => 1, i => 1] = 11.0
    B[i' => 2, i => 2] = 22.0

    enabled = ITensors.enable_threaded_blocksparse(false)
    C1 = A' * B
    ITensors.enable_threaded_blocksparse(true)
    C2 = A' * B
    ITensors.enable_threaded_blocksparse(enabled)

    @test nnzblocks(C1) == 0
    @test nnzblocks(C2) == 0
    @test nnz(C1) == 0
    @test nnz(C2) == 0
    @test C1 ≈ C2
  end

  BLAS.set_num_threads(blas_num_threads)
  ITensors.NDTensors.Strided.set_num_threads(strided_num_threads)
end
