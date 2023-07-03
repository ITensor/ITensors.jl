using NDTensors
using LinearAlgebra
using Test
if "cuda" in ARGS || "all" in ARGS
  using CUDA
end
if "metal" in ARGS || "all" in ARGS
  using Metal
end

@testset "BlockSparseTensor basic functionality" begin
  C = nothing
  include("device_list.jl")
  devs = devices_list(copy(ARGS))

  @testset "test device: $dev" for dev in devs
    # Indices
    indsA = ([2, 3], [4, 5])

    # Locations of non-zero blocks
    locs = [(1, 2), (2, 1)]

    A = dev(BlockSparseTensor(locs, indsA...))
    randn!(A)

    @test blockdims(A, (1, 2)) == (2, 5)
    @test blockdims(A, (2, 1)) == (3, 4)
    @test nnzblocks(A) == 2
    @test nnz(A) == 2 * 5 + 3 * 4
    @test inds(A) == ([2, 3], [4, 5])
    @test isblocknz(A, (2, 1))
    @test isblocknz(A, (1, 2))
    @test !isblocknz(A, (1, 1))
    @test !isblocknz(A, (2, 2))

    # Test different ways of getting nnz
    @test nnz(blockoffsets(A), inds(A)) == nnz(A)

    B = 2 * A
    @test B[1, 1] == 2 * A[1, 1]
    @test nnz(A) == 2 * 5 + 3 * 4
    @test nnz(B) == 2 * 5 + 3 * 4
    @test nnzblocks(A) == 2
    @test nnzblocks(B) == 2

    B = A / 2
    @test B[1, 1] == A[1, 1] / 2
    @test nnz(A) == 2 * 5 + 3 * 4
    @test nnz(B) == 2 * 5 + 3 * 4
    @test nnzblocks(A) == 2
    @test nnzblocks(B) == 2

    A[1, 5] = 15
    A[2, 5] = 25

    @test A[1, 1] == 0
    @test A[1, 5] == 15
    @test A[2, 5] == 25

    D = dense(A)

    @test D == A

    for I in eachindex(A)
      @test D[I] == A[I]
    end

    A12 = blockview(A, (1, 2))

    @test dims(A12) == (2, 5)

    for I in eachindex(A12)
      @test A12[I] == A[I + CartesianIndex(0, 4)]
    end

    B = dev(BlockSparseTensor(undef, locs, indsA))
    randn!(B)

    C = A + B

    for I in eachindex(C)
      @test C[I] == A[I] + B[I]
    end

    Ap = permutedims(A, (2, 1))

    @test blockdims(Ap, (1, 2)) == (4, 3)
    @test blockdims(Ap, (2, 1)) == (5, 2)
    @test nnz(A) == nnz(Ap)
    @test nnzblocks(A) == nnzblocks(Ap)

    for I in eachindex(C)
      @test A[I] == Ap[NDTensors.permute(I, (2, 1))]
    end

    A = dev(BlockSparseTensor(ComplexF64, locs, indsA))
    randn!(A)
    @test conj(data(store(A))) == data(store(conj(A)))
    @test typeof(conj(A)) <: BlockSparseTensor

    @testset "Random constructor" begin
      T = dev(randomBlockSparseTensor([(1, 1), (2, 2)], ([2, 2], [2, 2])))
      @test nnzblocks(T) == 2
      @test nnz(T) == 8
      @test eltype(T) == Float64
      @test norm(T) ≉ 0

      Tc = dev(randomBlockSparseTensor(ComplexF64, [(1, 1), (2, 2)], ([2, 2], [2, 2])))
      @test nnzblocks(Tc) == 2
      @test nnz(Tc) == 8
      @test eltype(Tc) == ComplexF64
      @test norm(Tc) ≉ 0
    end

    @testset "Complex Valued Operations" begin
      T = dev(randomBlockSparseTensor(ComplexF64, [(1, 1), (2, 2)], ([2, 2], [2, 2])))
      rT = real(T)
      @test eltype(rT) == Float64
      @test nnzblocks(rT) == nnzblocks(T)
      iT = imag(T)
      @test eltype(iT) == Float64
      @test nnzblocks(iT) == nnzblocks(T)
      @test norm(rT)^2 + norm(iT)^2 ≈ norm(T)^2

      cT = conj(T)
      @test eltype(cT) == ComplexF64
      @test nnzblocks(cT) == nnzblocks(T)
    end
    @testset "similartype regression test" begin
      # Regression test for issue seen in:
      # https://github.com/ITensor/ITensorInfiniteMPS.jl/pull/77
      # Previously, `similartype` wasn't using information about the dimensions
      # properly and was returning a `BlockSparse` storage of the dimensions
      # of the input tensor.
      T = dev(BlockSparseTensor([(1, 1)], ([2], [2])))
      @test NDTensors.ndims(
        NDTensors.storagetype(NDTensors.similartype(typeof(T), ([2], [2], [2])))
      ) == 3
    end

    @testset "Random constructor" begin
      T = dev(randomBlockSparseTensor([(1, 1), (2, 2)], ([2, 2], [2, 2])))
      @test nnzblocks(T) == 2
      @test nnz(T) == 8
      @test eltype(T) == Float64
      @test norm(T) ≉ 0

      Tc = dev(randomBlockSparseTensor(ComplexF64, [(1, 1), (2, 2)], ([2, 2], [2, 2])))
      @test nnzblocks(Tc) == 2
      @test nnz(Tc) == 8
      @test eltype(Tc) == ComplexF64
      @test norm(Tc) ≉ 0
    end

    @testset "permute_combine" begin
      indsA = ([2, 3], [4, 5], [6, 7, 8])
      locsA = [(2, 1, 1), (1, 2, 1), (2, 2, 3)]
      A = dev(BlockSparseTensor(locsA, indsA...))
      randn!(A)

      B = NDTensors.permute_combine(A, 3, (2, 1))
      @test nnzblocks(A) == nnzblocks(B)
      @test nnz(A) == nnz(B)

      Ap = permutedims(A, (3, 2, 1))

      for (bAp, bB) in zip(eachnzblock(Ap), eachnzblock(B))
        blockAp = blockview(Ap, bAp)
        blockB = blockview(B, bB)
        @test reshape(blockAp, size(blockB)) == blockB
      end
    end
  end

  @testset "BlockSparseTensor setindex! add block" begin
    T = BlockSparseTensor([2, 3], [4, 5])

    for I in eachindex(C)
      @test T[I] == 0.0
    end
    @test nnz(T) == 0
    @test nnzblocks(T) == 0
    @test !isblocknz(T, (1, 1))
    @test !isblocknz(T, (2, 1))
    @test !isblocknz(T, (1, 2))
    @test !isblocknz(T, (2, 2))

    T[1, 1] = 1.0

    @test T[1, 1] == 1.0
    @test nnz(T) == 8
    @test nnzblocks(T) == 1
    @test isblocknz(T, (1, 1))
    @test !isblocknz(T, (2, 1))
    @test !isblocknz(T, (1, 2))
    @test !isblocknz(T, (2, 2))

    T[4, 8] = 2.0

    @test T[4, 8] == 2.0
    @test nnz(T) == 8 + 15
    @test nnzblocks(T) == 2
    @test isblocknz(T, (1, 1))
    @test !isblocknz(T, (2, 1))
    @test !isblocknz(T, (1, 2))
    @test isblocknz(T, (2, 2))

    T[1, 6] = 3.0

    @test T[1, 6] == 3.0
    @test nnz(T) == 8 + 15 + 10
    @test nnzblocks(T) == 3
    @test isblocknz(T, (1, 1))
    @test !isblocknz(T, (2, 1))
    @test isblocknz(T, (1, 2))
    @test isblocknz(T, (2, 2))

    T[4, 2] = 4.0

    @test T[4, 2] == 4.0
    @test nnz(T) == 8 + 15 + 10 + 12
    @test nnzblocks(T) == 4
    @test isblocknz(T, (1, 1))
    @test isblocknz(T, (2, 1))
    @test isblocknz(T, (1, 2))
    @test isblocknz(T, (2, 2))
  end

  @testset "svd" begin
    @testset "svd example 1" begin
      A = BlockSparseTensor([(2, 1), (1, 2)], [2, 2], [2, 2])
      randn!(A)
      U, S, V = svd(A)
      @test isapprox(norm(array(U) * array(S) * array(V)' - array(A)), 0; atol=1e-14)
    end

    @testset "svd example 2" begin
      A = BlockSparseTensor([(1, 2), (2, 3)], [2, 2], [3, 2, 3])
      randn!(A)
      U, S, V = svd(A)
      @test isapprox(norm(array(U) * array(S) * array(V)' - array(A)), 0.0; atol=1e-14)
    end

    @testset "svd example 3" begin
      A = BlockSparseTensor([(2, 1), (3, 2)], [3, 2, 3], [2, 2])
      randn!(A)
      U, S, V = svd(A)
      @test isapprox(norm(array(U) * array(S) * array(V)' - array(A)), 0.0; atol=1e-14)
    end

    @testset "svd example 4" begin
      A = BlockSparseTensor([(2, 1), (3, 2)], [2, 3, 4], [5, 6])
      randn!(A)
      U, S, V = svd(A)
      @test isapprox(norm(array(U) * array(S) * array(V)' - array(A)), 0.0; atol=1e-13)
    end

    @testset "svd example 5" begin
      A = BlockSparseTensor([(1, 2), (2, 3)], [5, 6], [2, 3, 4])
      randn!(A)
      U, S, V = svd(A)
      @test isapprox(norm(array(U) * array(S) * array(V)' - array(A)), 0.0; atol=1e-13)
    end
  end

  @testset "exp" begin
    A = BlockSparseTensor([(1, 1), (2, 2)], [2, 4], [2, 4])
    randn!(A)
    expT = exp(A)
    @test isapprox(norm(array(expT) - exp(array(A))), 0.0; atol=1e-13)

    # Hermitian case
    A = BlockSparseTensor(ComplexF64, [(1, 1), (2, 2)], ([2, 2], [2, 2]))
    randn!(A)
    Ah = BlockSparseTensor(ComplexF64, undef, [(1, 1), (2, 2)], ([2, 2], [2, 2]))
    for bA in eachnzblock(A)
      b = blockview(A, bA)
      blockview(Ah, bA) .= b + b'
    end
    expTh = exp(Hermitian(Ah))
    @test array(expTh) ≈ exp(Hermitian(array(Ah))) rtol = 1e-13

    A = BlockSparseTensor([(2, 1), (1, 2)], [2, 2], [2, 2])
    @test_throws ErrorException exp(A)
  end
end

nothing
