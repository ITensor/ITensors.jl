@eval module $(gensym())
using Dictionaries: Dictionary
using GPUArraysCore: @allowscalar
using LinearAlgebra: norm
using NDTensors:
    NDTensors,
    Block,
    BlockSparseTensor,
    Diag,
    DiagBlockSparse,
    Tensor,
    blockoffsets,
    contract,
    dense,
    denseblocks,
    inds,
    nzblocks
using Random: randn!
using Test: @test, @test_broken, @test_throws, @testset
@testset "UniformDiagBlockSparseTensor basic functionality" begin
    NeverAlias = NDTensors.NeverAlias
    AllowAlias = NDTensors.AllowAlias

    storage = DiagBlockSparse(1.0, Dictionary([Block(1, 1), Block(2, 2)], [0, 1]))
    tensor = Tensor(storage, ([1, 1], [1, 1]))

    @test conj(tensor) == tensor
    @test conj(NeverAlias(), tensor) == tensor
    @test conj(AllowAlias(), tensor) == tensor

    c = 1 + 2im
    tensor *= c

    @test tensor[1, 1] == c
    @test conj(tensor) ≠ tensor
    @test conj(NeverAlias(), tensor) ≠ tensor
    @test conj(AllowAlias(), tensor) ≠ tensor
    @test conj(tensor)[1, 1] == conj(c)
    @test conj(NeverAlias(), tensor)[1, 1] == conj(c)
    @test conj(AllowAlias(), tensor)[1, 1] == conj(c)
end
@testset "DiagBlockSparse off-diagonal (eltype=$elt)" for elt in (
        Float32, Float64, Complex{Float32}, Complex{Float64},
    )
    inds1 = ([1, 1], [1, 1])
    inds2 = ([1, 1], [1, 1])
    blocks = [(1, 2), (2, 1)]
    a1 = BlockSparseTensor{elt}(blocks, inds1...)
    for b in nzblocks(a1)
        randn!(a1[b])
    end
    a2 = Tensor(DiagBlockSparse(one(elt), blockoffsets(a1)), inds2)
    for (labels1, labels2) in (((1, -1), (-1, 2)), ((-1, -2), (-1, -2)))
        @test_throws ErrorException contract(a1, labels1, a2, labels2)
    end
end

include("NDTensorsTestUtils/NDTensorsTestUtils.jl")
using .NDTensorsTestUtils: devices_list
@testset "DiagBlockSparse contract" for dev in devices_list(copy(ARGS))
    elt = dev == NDTensors.mtl ? Float32 : Float64
    A = dev(BlockSparseTensor{elt}([(1, 1), (2, 2)], [2, 2], [2, 2]))
    randn!(A)
    t = Tensor(DiagBlockSparse(one(elt), blockoffsets(A)), inds(A))
    tdense = Tensor(Diag(one(elt)), inds(A))

    a = dense(contract(A, (1, -2), t, (3, -2)))
    b = contract(dense(A), (1, -2), tdense, (3, -2))
    @test @allowscalar a ≈ b

    a = dense(contract(A, (-2, 1), t, (-2, 3)))
    b = contract(dense(A), (-2, 1), tdense, (-2, 3))
    @test @allowscalar a ≈ b

    a = contract(A, (-1, -2), t, (-1, -2))[]
    b = contract(dense(A), (-1, -2), tdense, (-1, -2))[]
    @test @allowscalar a ≈ b

    ## TODO fix these kinds of contractions
    A = BlockSparseTensor{elt}([(1, 1), (2, 2)], [3, 2, 3], [2, 2])
    randn!(A)
    t = Tensor(DiagBlockSparse(one(elt), blockoffsets(A)), inds(A))
    @test dense(contract(A, (1, -2), (t), (3, -2))) ≈
        contract(dense(A), (1, -2), dense(t), (3, -2))
    @test dense(contract(A, (-2, 1), t, (-2, 3))) ≈
        contract(dense(A), (-2, 1), dense(t), (-2, 3))
    @test contract(dev(A), (-1, -2), dev(t), (-1, -2))[] ≈
        contract(dense(A), (-1, -2), dense(t), (-1, -2))[]
end

@testset "UniformDiagBlockSparse norm" begin
    elt = Float64
    storage = DiagBlockSparse(one(elt), Dictionary([Block(1, 1), Block(2, 2)], [0, 2]))
    tensor = Tensor(storage, ([2, 2], [2, 2]))
    @test norm(tensor) ≈ norm(dense(tensor))

    elt = Float64
    storage = DiagBlockSparse(one(elt), Dictionary([Block(1, 1)], [0]))
    tensor = Tensor(storage, ([2], [1, 1]))
    @test norm(tensor) ≈ norm(dense(tensor))
end

@testset "DiagBlockSparse denseblocks" begin
    elt = Float64
    blockoffsets_a = Dictionary([Block(1, 1), Block(2, 2)], [0, 2])
    inds_a = ([2, 2], [2, 2])
    a = Tensor(DiagBlockSparse(elt, blockoffsets_a, 4), inds_a)
    a[Block(1, 1)][1, 1] = 1
    a[Block(1, 1)][2, 2] = 2
    a[Block(2, 2)][1, 1] = 3
    a[Block(2, 2)][2, 2] = 4
    a′ = denseblocks(a)
    @test dense(a) == dense(a′)

    elt = Float64
    blockoffsets_a = Dictionary([Block(1, 1)], [0])
    inds_a = ([2], [1, 1])
    a = Tensor(DiagBlockSparse(one(elt), blockoffsets_a), inds_a)
    a′ = denseblocks(a)
    @test dense(a) == dense(a′)
end

end
