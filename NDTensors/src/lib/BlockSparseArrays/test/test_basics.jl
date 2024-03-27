@eval module $(gensym())
using BlockArrays: Block, BlockedUnitRange, blockedrange, blocklength, blocksize
using LinearAlgebra: mul!
using NDTensors.BlockSparseArrays: BlockSparseArray, block_nstored, block_reshape
using NDTensors.SparseArrayInterface: nstored
using NDTensors.TensorAlgebra: contract
using Test: @test, @testset, @test_broken
include("TestBlockSparseArraysUtils.jl")
@testset "BlockSparseArrays (eltype=$elt)" for elt in
                                               (Float32, Float64, ComplexF32, ComplexF64)
  @testset "Basics" begin
    a = BlockSparseArray{elt}([2, 3], [2, 3])
    @test a == BlockSparseArray{elt}(blockedrange([2, 3]), blockedrange([2, 3]))
    @test eltype(a) === elt
    @test axes(a) == (1:5, 1:5)
    @test all(aᵢ -> aᵢ isa BlockedUnitRange, axes(a))
    @test blocklength.(axes(a)) == (2, 2)
    @test blocksize(a) == (2, 2)
    @test size(a) == (5, 5)
    @test block_nstored(a) == 0
    @test iszero(a)
    @test all(I -> iszero(a[I]), eachindex(a))

    a = BlockSparseArray{elt}([2, 3], [2, 3])
    a[3, 3] = 33
    @test eltype(a) === elt
    @test axes(a) == (1:5, 1:5)
    @test all(aᵢ -> aᵢ isa BlockedUnitRange, axes(a))
    @test blocklength.(axes(a)) == (2, 2)
    @test blocksize(a) == (2, 2)
    @test size(a) == (5, 5)
    @test block_nstored(a) == 1
    @test !iszero(a)
    @test a[3, 3] == 33
    @test all(eachindex(a)) do I
      if I == CartesianIndex(3, 3)
        a[I] == 33
      else
        iszero(a[I])
      end
    end
  end
  @testset "Tensor algebra" begin
    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    a[Block(1, 2)] = randn(elt, size(@view(a[Block(1, 2)])))
    a[Block(2, 1)] = randn(elt, size(@view(a[Block(2, 1)])))
    @test eltype(a) == elt
    @test block_nstored(a) == 2
    @test nstored(a) == 2 * 4 + 3 * 3

    b = similar(a, complex(elt))
    @test eltype(b) == complex(eltype(a))
    @test iszero(b)
    @test block_nstored(b) == 0
    @test nstored(b) == 0
    @test size(b) == size(a)
    @test blocksize(b) == blocksize(a)

    b = copy(a)
    b[1, 1] = 11
    @test b[1, 1] == 11
    @test a[1, 1] ≠ 11

    b = copy(a)
    b .*= 2
    @test b ≈ 2a

    b = copy(a)
    b ./= 2
    @test b ≈ a / 2

    b = 2 * a
    @test Array(b) ≈ 2 * Array(a)
    @test eltype(b) == elt
    @test block_nstored(b) == 2
    @test nstored(b) == 2 * 4 + 3 * 3

    b = (2 + 3im) * a
    @test Array(b) ≈ (2 + 3im) * Array(a)
    @test eltype(b) == complex(elt)
    @test block_nstored(b) == 2
    @test nstored(b) == 2 * 4 + 3 * 3

    b = a + a
    @test Array(b) ≈ 2 * Array(a)
    @test eltype(b) == elt
    @test block_nstored(b) == 2
    @test nstored(b) == 2 * 4 + 3 * 3

    x = BlockSparseArray{elt}(undef, ([3, 4], [2, 3]))
    x[Block(1, 2)] = randn(elt, size(@view(x[Block(1, 2)])))
    x[Block(2, 1)] = randn(elt, size(@view(x[Block(2, 1)])))
    b = a .+ a .+ 3 .* PermutedDimsArray(x, (2, 1))
    @test Array(b) ≈ 2 * Array(a) + 3 * permutedims(Array(x), (2, 1))
    @test eltype(b) == elt
    @test block_nstored(b) == 2
    @test nstored(b) == 2 * 4 + 3 * 3

    b = permutedims(a, (2, 1))
    @test Array(b) ≈ permutedims(Array(a), (2, 1))
    @test eltype(b) == elt
    @test block_nstored(b) == 2
    @test nstored(b) == 2 * 4 + 3 * 3

    b = map(x -> 2x, a)
    @test Array(b) ≈ 2 * Array(a)
    @test eltype(b) == elt
    @test size(b) == size(a)
    @test blocksize(b) == (2, 2)
    @test block_nstored(b) == 2
    @test nstored(b) == 2 * 4 + 3 * 3

    b = a[[Block(2), Block(1)], [Block(2), Block(1)]]
    @test b[Block(1, 1)] == a[Block(2, 2)]
    @test b[Block(1, 2)] == a[Block(2, 1)]
    @test b[Block(2, 1)] == a[Block(1, 2)]
    @test b[Block(2, 2)] == a[Block(1, 1)]
    @test size(b) == size(a)
    @test blocksize(b) == (2, 2)
    @test nstored(b) == nstored(a)
    @test block_nstored(b) == 2

    b = a[Block(1):Block(2), Block(1):Block(2)]
    @test b == a
    @test size(b) == size(a)
    @test blocksize(b) == (2, 2)
    @test nstored(b) == nstored(a)
    @test block_nstored(b) == 2

    b = a[Block(1):Block(1), Block(1):Block(2)]
    @test b == Array(a)[1:2, 1:end]
    @test b[Block(1, 1)] == a[Block(1, 1)]
    @test b[Block(1, 2)] == a[Block(1, 2)]
    @test size(b) == (2, 7)
    @test blocksize(b) == (1, 2)
    @test nstored(b) == nstored(a[Block(1, 2)])
    @test block_nstored(b) == 1

    b = a[2:4, 2:4]
    @test b == Array(a)[2:4, 2:4]
    @test size(b) == (3, 3)
    @test blocksize(b) == (2, 2)
    @test nstored(b) == 1 * 1 + 2 * 2
    @test block_nstored(b) == 2

    b = a[Block(2, 1)[1:2, 2:3]]
    @test b == Array(a)[3:4, 2:3]
    @test size(b) == (2, 2)
    @test blocksize(b) == (1, 1)
    @test nstored(b) == 2 * 2
    @test block_nstored(b) == 1

    ## Broken, need to fix.

    @test_broken a[Block(1), Block(1):Block(2)]

    # This is outputting only zero blocks.
    b = a[Block(2):Block(2), Block(1):Block(2)]
    @test_broken block_nstored(b) == 1
    @test_broken b == Array(a)[3:5, 1:end]

    b = a'
    @test_broken block_nstored(b) == 2

    b = transpose(a)
    @test_broken block_nstored(b) == 2

    b = copy(a)
    x = randn(size(@view(a[Block(2, 2)])))
    b[Block(2), Block(2)] = x
    @test_broken b[Block(2, 2)] == x

    # Doesnt' set the block
    b = copy(a)
    b[Block(1, 1)] .= 1
    @test_broken b[1, 1] == trues(size(@view(b[1, 1])))
  end
  @testset "LinearAlgebra" begin
    a1 = BlockSparseArray{elt}([2, 3], [2, 3])
    a1[Block(1, 1)] = randn(elt, size(@view(a1[Block(1, 1)])))
    a2 = BlockSparseArray{elt}([2, 3], [2, 3])
    a2[Block(1, 1)] = randn(elt, size(@view(a1[Block(1, 1)])))
    a_dest = a1 * a2
    @test Array(a_dest) ≈ Array(a1) * Array(a2)
    @test a_dest isa BlockSparseArray{elt}
    @test block_nstored(a_dest) == 1
  end
  @testset "TensorAlgebra" begin
    a1 = BlockSparseArray{elt}([2, 3], [2, 3])
    a1[Block(1, 1)] = randn(elt, size(@view(a1[Block(1, 1)])))
    a2 = BlockSparseArray{elt}([2, 3], [2, 3])
    a2[Block(1, 1)] = randn(elt, size(@view(a1[Block(1, 1)])))
    # TODO: Make this work, requires customization of `TensorAlgebra.fusedims` and
    # `TensorAlgebra.splitdims` in terms of `BlockSparseArrays.block_reshape`,
    # and customization of `TensorAlgebra.:⊗` in terms of `GradedAxes.tensor_product`.
    a_dest, dimnames_dest = contract(a1, (1, -1), a2, (-1, 2))
    a_dest_dense, dimnames_dest_dense = contract(Array(a1), (1, -1), Array(a2), (-1, 2))
    @test a_dest ≈ a_dest_dense
  end
  @testset "block_reshape" begin
    a = BlockSparseArray{elt}(undef, ([3, 4], [2, 3]))
    a[Block(1, 2)] = randn(elt, size(@view(a[Block(1, 2)])))
    a[Block(2, 1)] = randn(elt, size(@view(a[Block(2, 1)])))
    b = block_reshape(a, [6, 8, 9, 12])
    @test reshape(a[Block(1, 2)], 9) == b[Block(3)]
    @test reshape(a[Block(2, 1)], 8) == b[Block(2)]
    @test block_nstored(b) == 2
    @test nstored(b) == 17
  end
end
end
