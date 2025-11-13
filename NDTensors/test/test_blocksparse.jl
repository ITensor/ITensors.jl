@eval module $(gensym())
using GPUArraysCore: @allowscalar
using LinearAlgebra: Hermitian, exp, norm, svd
using NDTensors:
    NDTensors,
    BlockSparseTensor,
    array,
    blockdims,
    blockoffsets,
    blockview,
    data,
    dense,
    diag,
    diaglength,
    dims,
    eachnzblock,
    inds,
    isblocknz,
    nnz,
    nnzblocks,
    randomBlockSparseTensor,
    store,
    storage
include("NDTensorsTestUtils/NDTensorsTestUtils.jl")
using .NDTensorsTestUtils: default_rtol, devices_list, is_supported_eltype
using Random: randn!
using Test: @test, @test_throws, @testset

@testset "BlockSparseTensor basic functionality" begin
    C = nothing

    @testset "test device: $dev, eltype: $elt" for dev in devices_list(copy(ARGS)),
            elt in (Float32, Float64)

        if !is_supported_eltype(dev, elt)
            continue
        end
        # Indices
        indsA = ([2, 3], [4, 5])

        # Locations of non-zero blocks
        locs = [(1, 2), (2, 1)]

        A = dev(BlockSparseTensor{elt}(locs, indsA...))
        randn!(A)

        @test blockdims(A, (1, 2)) == (2, 5)
        @test blockdims(A, (2, 1)) == (3, 4)
        @test !isempty(A)
        @test nnzblocks(A) == 2
        @test nnz(A) == 2 * 5 + 3 * 4
        @test inds(A) == ([2, 3], [4, 5])
        @test isblocknz(A, (2, 1))
        @test isblocknz(A, (1, 2))
        @test !isblocknz(A, (1, 1))
        @test !isblocknz(A, (2, 2))
        dA = diag(A)
        @test @allowscalar dA ≈ diag(dense(A))
        @test sum(A) ≈ sum(array(A))
        @test prod(A) ≈ prod(array(A))

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

        @allowscalar begin
            A[1, 5] = 15
            A[2, 5] = 25

            @test A[1, 1] == 0
            @test A[1, 5] == 15
            @test A[2, 5] == 25
        end
        D = dense(A)

        @allowscalar begin
            @test D == A

            for I in eachindex(A)
                @test D[I] == A[I]
            end
        end

        A12 = blockview(A, (1, 2))

        @test dims(A12) == (2, 5)

        @allowscalar for I in eachindex(A12)
            @test A12[I] == A[I + CartesianIndex(0, 4)]
        end

        B = dev(BlockSparseTensor(elt, undef, locs, indsA))
        randn!(B)

        C = A + B

        @allowscalar for I in eachindex(C)
            @test C[I] == A[I] + B[I]
        end
        Cp = NDTensors.map_diag(i -> 2 * i, C)
        @allowscalar for i in 1:diaglength(Cp)
            @test Cp[i, i] == 2 * C[i, i]
        end

        Ap = permutedims(A, (2, 1))

        @test blockdims(Ap, (1, 2)) == (4, 3)
        @test blockdims(Ap, (2, 1)) == (5, 2)
        @test nnz(A) == nnz(Ap)
        @test nnzblocks(A) == nnzblocks(Ap)

        @allowscalar for I in eachindex(C)
            @test A[I] == Ap[NDTensors.permute(I, (2, 1))]
        end

        A = dev(BlockSparseTensor(complex(elt), locs, indsA))
        randn!(A)
        @test conj(data(store(A))) == data(store(conj(A)))
        @test typeof(conj(A)) <: BlockSparseTensor

        @testset "No blocks" begin
            T = dev(BlockSparseTensor{elt}(Tuple{Int, Int}[], [2, 2], [2, 2]))
            @test nnzblocks(T) == 0
            @test size(T) == (4, 4)
            @test length(T) == 16
            @test !isempty(T)
            @test isempty(storage(T))
            @test nnz(T) == 0
            @test eltype(T) == elt
            @test norm(T) == 0
        end

        @testset "Empty" begin
            T = dev(BlockSparseTensor{elt}(Tuple{Int, Int}[], Int[], Int[]))
            @test nnzblocks(T) == 0
            @test size(T) == (0, 0)
            @test length(T) == 0
            @test isempty(T)
            @test isempty(storage(T))
            @test nnz(T) == 0
            @test eltype(T) == elt
            @test norm(T) == 0
        end

        @testset "Random constructor" begin
            T = dev(randomBlockSparseTensor(elt, [(1, 1), (2, 2)], ([2, 2], [2, 2])))
            @test nnzblocks(T) == 2
            @test nnz(T) == 8
            @test eltype(T) == elt
            @test norm(T) ≉ 0

            Tc = dev(randomBlockSparseTensor(complex(elt), [(1, 1), (2, 2)], ([2, 2], [2, 2])))
            @test nnzblocks(Tc) == 2
            @test nnz(Tc) == 8
            @test eltype(Tc) == complex(elt)
            @test norm(Tc) ≉ 0
        end

        @testset "Complex Valued Operations" begin
            T = dev(randomBlockSparseTensor(complex(elt), [(1, 1), (2, 2)], ([2, 2], [2, 2])))
            rT = real(T)
            @test eltype(rT) == elt
            @test nnzblocks(rT) == nnzblocks(T)
            iT = imag(T)
            @test eltype(iT) == elt
            @test nnzblocks(iT) == nnzblocks(T)
            @test norm(rT)^2 + norm(iT)^2 ≈ norm(T)^2

            cT = conj(T)
            @test eltype(cT) == complex(elt)
            @test nnzblocks(cT) == nnzblocks(T)
        end

        @testset "similartype regression test" begin
            # Regression test for issue seen in:
            # https://github.com/ITensor/ITensorInfiniteMPS.jl/pull/77
            # Previously, `similartype` wasn't using information about the dimensions
            # properly and was returning a `BlockSparse` storage of the dimensions
            # of the input tensor.
            T = dev(BlockSparseTensor(elt, [(1, 1)], ([2], [2])))
            @test NDTensors.ndims(
                NDTensors.storagetype(NDTensors.similartype(typeof(T), ([2], [2], [2])))
            ) == 3
        end

        @testset "Random constructor" begin
            T = dev(randomBlockSparseTensor(elt, [(1, 1), (2, 2)], ([2, 2], [2, 2])))
            @test nnzblocks(T) == 2
            @test nnz(T) == 8
            @test eltype(T) == elt
            @test norm(T) ≉ 0

            Tc = dev(randomBlockSparseTensor(complex(elt), [(1, 1), (2, 2)], ([2, 2], [2, 2])))
            @test nnzblocks(Tc) == 2
            @test nnz(Tc) == 8
            @test eltype(Tc) == complex(elt)
            @test norm(Tc) ≉ 0
        end

        @testset "permute_combine" begin
            indsA = ([2, 3], [4, 5], [6, 7, 8])
            locsA = [(2, 1, 1), (1, 2, 1), (2, 2, 3)]
            A = dev(BlockSparseTensor{elt}(locsA, indsA...))
            randn!(A)

            B = NDTensors.permute_combine(A, 3, (2, 1))
            @test nnzblocks(A) == nnzblocks(B)
            @test nnz(A) == nnz(B)

            Ap = NDTensors.permutedims(A, (3, 2, 1))

            @allowscalar for (bAp, bB) in zip(eachnzblock(Ap), eachnzblock(B))
                blockAp = blockview(Ap, bAp)
                blockB = blockview(B, bB)
                @test reshape(blockAp, size(blockB)) == blockB
            end
        end
    end

    @testset "BlockSparseTensor setindex! add block" begin
        T = BlockSparseTensor([2, 3], [4, 5])

        @allowscalar for I in eachindex(T)
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

    @testset "svd on $dev, eltype: $elt" for dev in devices_list(copy(ARGS)),
            elt in (Float32, Float64)

        if !is_supported_eltype(dev, elt)
            continue
        end
        @testset "svd example 1" begin
            A = dev(BlockSparseTensor{elt}([(2, 1), (1, 2)], [2, 2], [2, 2]))
            randn!(A)
            U, S, V = svd(A)
            @test @allowscalar array(U) * array(S) * array(V)' ≈ array(A)
            atol = default_rtol(elt)
        end

        @testset "svd example 2" begin
            A = dev(BlockSparseTensor{elt}([(1, 2), (2, 3)], [2, 2], [3, 2, 3]))
            randn!(A)
            U, S, V = svd(A)
            @test @allowscalar array(U) * array(S) * array(V)' ≈ array(A)
            atol = default_rtol(elt)
        end

        @testset "svd example 3" begin
            A = dev(BlockSparseTensor{elt}([(2, 1), (3, 2)], [3, 2, 3], [2, 2]))
            randn!(A)
            U, S, V = svd(A)
            @test @allowscalar array(U) * array(S) * array(V)' ≈ array(A)
            atol = default_rtol(elt)
        end

        @testset "svd example 4" begin
            A = dev(BlockSparseTensor{elt}([(2, 1), (3, 2)], [2, 3, 4], [5, 6]))
            randn!(A)
            U, S, V = svd(A)
            @test @allowscalar array(U) * array(S) * array(V)' ≈ array(A)
            atol = default_rtol(elt)
        end

        @testset "svd example 5" begin
            A = dev(BlockSparseTensor{elt}([(1, 2), (2, 3)], [5, 6], [2, 3, 4]))
            randn!(A)
            U, S, V = svd(A)
            @test @allowscalar array(U) * array(S) * array(V)' ≈ array(A)
            atol = default_rtol(elt)
        end
    end

    @testset "exp, eltype: $elt" for elt in (Float32, Float64)
        A = BlockSparseTensor{elt}([(1, 1), (2, 2)], [2, 4], [2, 4])
        randn!(A)
        expT = exp(A)
        @test array(expT) ≈ exp(array(A))
        atol = default_rtol(elt)

        # Hermitian case
        A = BlockSparseTensor(complex(elt), [(1, 1), (2, 2)], ([2, 2], [2, 2]))
        randn!(A)
        Ah = BlockSparseTensor(complex(elt), undef, [(1, 1), (2, 2)], ([2, 2], [2, 2]))
        for bA in eachnzblock(A)
            b = blockview(A, bA)
            blockview(Ah, bA) .= b + b'
        end
        expTh = exp(Hermitian(Ah))
        @test array(expTh) ≈ exp(Hermitian(array(Ah))) rtol = default_rtol(eltype(Ah))

        A = BlockSparseTensor{elt}([(2, 1), (1, 2)], [2, 2], [2, 2])
        @test_throws ErrorException exp(A)
    end
end
end
