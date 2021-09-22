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

    if using_threaded_blocksparse
      ITensors.enable_threaded_blocksparse()
    else
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
    if using_threaded_blocksparse
      ITensors.enable_threaded_blocksparse()
    else
      ITensors.disable_threaded_blocksparse()
    end

    @test nnzblocks(C1) == 0
    @test nnzblocks(C2) == 0
    @test nnz(C1) == 0
    @test nnz(C2) == 0
    @test C1 ≈ C2
  end

  BLAS.set_num_threads(blas_num_threads)
  ITensors.NDTensors.Strided.set_num_threads(strided_num_threads)
end
