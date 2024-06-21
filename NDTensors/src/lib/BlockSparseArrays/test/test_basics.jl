@eval module $(gensym())
using BlockArrays:
  Block,
  BlockIndexRange,
  BlockRange,
  BlockSlice,
  BlockVector,
  BlockedOneTo,
  BlockedUnitRange,
  blockedrange,
  blocklength,
  blocklengths,
  blocksize,
  blocksizes,
  mortar
using Compat: @compat
using LinearAlgebra: mul!
using NDTensors.BlockSparseArrays:
  @view!, BlockSparseArray, block_nstored, block_reshape, view!
using NDTensors.SparseArrayInterface: nstored
using NDTensors.TensorAlgebra: contract
using Test: @test, @test_broken, @test_throws, @testset
include("TestBlockSparseArraysUtils.jl")
@testset "BlockSparseArrays (eltype=$elt)" for elt in
                                               (Float32, Float64, ComplexF32, ComplexF64)
  @testset "Broken" begin
    a = BlockSparseArray{elt}([2, 3], [3, 4])
    b = @view a[[Block(2), Block(1)], [Block(2), Block(1)]]
    @test b isa SubArray{<:Any,<:Any,<:BlockSparseArray}
    @test_broken b[2:4, 2:4]

    a = BlockSparseArray{elt}([2, 3], [3, 4])
    b = @views a[[Block(2), Block(1)], [Block(2), Block(1)]][Block(1, 1)]
    @test_broken b isa SubArray{<:Any,<:Any,<:BlockSparseArray}

    a = BlockSparseArray{elt}([2, 3], [3, 4])
    b = @views a[Block(1, 1)][1:2, 1:1]
    @test b isa SubArray{<:Any,<:Any,<:BlockSparseArray}
    for i in parentindices(b)
      @test_broken i isa BlockSlice{<:BlockIndexRange{1}}
    end
  end
  @testset "Basics" begin
    a = BlockSparseArray{elt}([2, 3], [2, 3])
    @test a == BlockSparseArray{elt}(blockedrange([2, 3]), blockedrange([2, 3]))
    @test eltype(a) === elt
    @test axes(a) == (1:5, 1:5)
    @test all(aᵢ -> aᵢ isa BlockedOneTo, axes(a))
    @test blocklength.(axes(a)) == (2, 2)
    @test blocksize(a) == (2, 2)
    @test size(a) == (5, 5)
    @test block_nstored(a) == 0
    @test iszero(a)
    @test all(I -> iszero(a[I]), eachindex(a))
    @test_throws DimensionMismatch a[Block(1, 1)] = randn(elt, 2, 3)

    a = BlockSparseArray{elt}([2, 3], [2, 3])
    a[3, 3] = 33
    @test eltype(a) === elt
    @test axes(a) == (1:5, 1:5)
    @test all(aᵢ -> aᵢ isa BlockedOneTo, axes(a))
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
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    @test eltype(a) == elt
    @test block_nstored(a) == 2
    @test nstored(a) == 2 * 4 + 3 * 3

    a = BlockSparseArray{elt}([2, 3], [3, 4])
    a[Block(1, 2)] .= 2
    @test eltype(a) == elt
    @test all(==(2), a[Block(1, 2)])
    @test iszero(a[Block(1, 1)])
    @test iszero(a[Block(2, 1)])
    @test iszero(a[Block(2, 2)])
    @test block_nstored(a) == 1
    @test nstored(a) == 2 * 4

    a = BlockSparseArray{elt}([2, 3], [3, 4])
    a[Block(1, 2)] .= 0
    @test eltype(a) == elt
    @test iszero(a[Block(1, 1)])
    @test iszero(a[Block(2, 1)])
    @test iszero(a[Block(1, 2)])
    @test iszero(a[Block(2, 2)])
    @test block_nstored(a) == 1
    @test nstored(a) == 2 * 4

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = similar(a, complex(elt))
    @test eltype(b) == complex(eltype(a))
    @test iszero(b)
    @test block_nstored(b) == 0
    @test nstored(b) == 0
    @test size(b) == size(a)
    @test blocksize(b) == blocksize(a)

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = copy(a)
    b[1, 1] = 11
    @test b[1, 1] == 11
    @test a[1, 1] ≠ 11

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = copy(a)
    b .*= 2
    @test b ≈ 2a

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = copy(a)
    b ./= 2
    @test b ≈ a / 2

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = 2 * a
    @test Array(b) ≈ 2 * Array(a)
    @test eltype(b) == elt
    @test block_nstored(b) == 2
    @test nstored(b) == 2 * 4 + 3 * 3

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = (2 + 3im) * a
    @test Array(b) ≈ (2 + 3im) * Array(a)
    @test eltype(b) == complex(elt)
    @test block_nstored(b) == 2
    @test nstored(b) == 2 * 4 + 3 * 3

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = a + a
    @test Array(b) ≈ 2 * Array(a)
    @test eltype(b) == elt
    @test block_nstored(b) == 2
    @test nstored(b) == 2 * 4 + 3 * 3

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    x = BlockSparseArray{elt}(undef, ([3, 4], [2, 3]))
    @views for b in [Block(1, 2), Block(2, 1)]
      x[b] = randn(elt, size(x[b]))
    end
    b = a .+ a .+ 3 .* PermutedDimsArray(x, (2, 1))
    @test Array(b) ≈ 2 * Array(a) + 3 * permutedims(Array(x), (2, 1))
    @test eltype(b) == elt
    @test block_nstored(b) == 2
    @test nstored(b) == 2 * 4 + 3 * 3

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = permutedims(a, (2, 1))
    @test Array(b) ≈ permutedims(Array(a), (2, 1))
    @test eltype(b) == elt
    @test block_nstored(b) == 2
    @test nstored(b) == 2 * 4 + 3 * 3

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = map(x -> 2x, a)
    @test Array(b) ≈ 2 * Array(a)
    @test eltype(b) == elt
    @test size(b) == size(a)
    @test blocksize(b) == (2, 2)
    @test block_nstored(b) == 2
    @test nstored(b) == 2 * 4 + 3 * 3

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = a[[Block(2), Block(1)], [Block(2), Block(1)]]
    @test b[Block(1, 1)] == a[Block(2, 2)]
    @test b[Block(1, 2)] == a[Block(2, 1)]
    @test b[Block(2, 1)] == a[Block(1, 2)]
    @test b[Block(2, 2)] == a[Block(1, 1)]
    @test size(b) == size(a)
    @test blocksize(b) == (2, 2)
    @test nstored(b) == nstored(a)
    @test block_nstored(b) == 2

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = a[Block(1):Block(2), Block(1):Block(2)]
    @test b == a
    @test size(b) == size(a)
    @test blocksize(b) == (2, 2)
    @test nstored(b) == nstored(a)
    @test block_nstored(b) == 2

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = a[Block(1):Block(1), Block(1):Block(2)]
    @test b == Array(a)[1:2, 1:end]
    @test b[Block(1, 1)] == a[Block(1, 1)]
    @test b[Block(1, 2)] == a[Block(1, 2)]
    @test size(b) == (2, 7)
    @test blocksize(b) == (1, 2)
    @test nstored(b) == nstored(a[Block(1, 2)])
    @test block_nstored(b) == 1

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = a[2:4, 2:4]
    @test b == Array(a)[2:4, 2:4]
    @test size(b) == (3, 3)
    @test blocksize(b) == (2, 2)
    @test nstored(b) == 1 * 1 + 2 * 2
    @test block_nstored(b) == 2

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = a[Block(2, 1)[1:2, 2:3]]
    @test b == Array(a)[3:4, 2:3]
    @test size(b) == (2, 2)
    @test blocksize(b) == (1, 1)
    @test nstored(b) == 2 * 2
    @test block_nstored(b) == 1

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = PermutedDimsArray(a, (2, 1))
    @test block_nstored(b) == 2
    @test Array(b) == permutedims(Array(a), (2, 1))
    c = 2 * b
    @test block_nstored(c) == 2
    @test Array(c) == 2 * permutedims(Array(a), (2, 1))

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = a'
    @test block_nstored(b) == 2
    @test Array(b) == Array(a)'
    c = 2 * b
    @test block_nstored(c) == 2
    @test Array(c) == 2 * Array(a)'

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = transpose(a)
    @test block_nstored(b) == 2
    @test Array(b) == transpose(Array(a))
    c = 2 * b
    @test block_nstored(c) == 2
    @test Array(c) == 2 * transpose(Array(a))

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = a[Block(1), Block(1):Block(2)]
    @test size(b) == (2, 7)
    @test blocksize(b) == (1, 2)
    @test b[Block(1, 1)] == a[Block(1, 1)]
    @test b[Block(1, 2)] == a[Block(1, 2)]

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = copy(a)
    x = randn(elt, size(@view(a[Block(2, 2)])))
    b[Block(2), Block(2)] = x
    @test b[Block(2, 2)] == x

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = copy(a)
    b[Block(1, 1)] .= 1
    @test b[Block(1, 1)] == trues(blocksizes(b)[1, 1])

    a = BlockSparseArray{elt}([2, 3], [3, 4])
    b = @view a[Block(2, 2)]
    @test size(b) == (3, 4)
    for i in parentindices(b)
      @test i isa BlockSlice{<:Block{1}}
    end
    @test parentindices(b)[1] == BlockSlice(Block(2), 3:5)
    @test parentindices(b)[2] == BlockSlice(Block(2), 4:7)

    a = BlockSparseArray{elt}([2, 3], [3, 4])
    b = @view a[Block(2, 2)[1:2, 2:2]]
    @test size(b) == (2, 1)
    for i in parentindices(b)
      @test i isa BlockSlice{<:BlockIndexRange{1}}
    end
    @test parentindices(b)[1] == BlockSlice(Block(2)[1:2], 3:4)
    @test parentindices(b)[2] == BlockSlice(Block(2)[2:2], 5:5)

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    x = randn(elt, 1, 2)
    @view(a[Block(2, 2)])[1:1, 1:2] = x
    @test a[Block(2, 2)][1:1, 1:2] == x
    @test @view(a[Block(2, 2)])[1:1, 1:2] == x
    @test a[3:3, 4:5] == x

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    x = randn(elt, 1, 2)
    @views a[Block(2, 2)][1:1, 1:2] = x
    @test a[Block(2, 2)][1:1, 1:2] == x
    @test @view(a[Block(2, 2)])[1:1, 1:2] == x
    @test a[3:3, 4:5] == x

    a = BlockSparseArray{elt}([2, 3], [2, 3])
    @views for b in [Block(1, 1), Block(2, 2)]
      a[b] = randn(elt, size(a[b]))
    end
    for I in (
      Block.(1:2),
      [Block(1), Block(2)],
      BlockVector([Block(1), Block(2)], [1, 1]),
      # TODO: This should merge blocks.
      BlockVector([Block(1), Block(2)], [2]),
    )
      b = @view a[I, I]
      for I in CartesianIndices(a)
        @test b[I] == a[I]
      end
      for block in BlockRange(a)
        @test b[block] == a[block]
      end
    end

    a = BlockSparseArray{elt}([2, 3], [2, 3])
    @views for b in [Block(1, 1), Block(2, 2)]
      # TODO: Use `blocksizes(a)[Int.(Tuple(b))...]` once available.
      a[b] = randn(elt, size(a[b]))
    end
    for I in (
      [Block(2), Block(1)],
      BlockVector([Block(2), Block(1)], [1, 1]),
      # TODO: This should merge blocks.
      BlockVector([Block(2), Block(1)], [2]),
    )
      b = @view a[I, I]
      @test b[Block(1, 1)] == a[Block(2, 2)]
      @test b[Block(2, 1)] == a[Block(1, 2)]
      @test b[Block(1, 2)] == a[Block(2, 1)]
      @test b[Block(2, 2)] == a[Block(1, 1)]
      @test b[1, 1] == a[3, 3]
      @test b[4, 4] == a[1, 1]
      b[4, 4] = 44
      @test b[4, 4] == 44
    end

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = a[Block(2):Block(2), Block(1):Block(2)]
    @test block_nstored(b) == 1
    @test b == Array(a)[3:5, 1:end]

    a = BlockSparseArray{elt}(undef, ([2, 3, 4], [2, 3, 4]))
    # TODO: Define `block_diagindices`.
    @views for b in [Block(1, 1), Block(2, 2), Block(3, 3)]
      a[b] = randn(elt, size(a[b]))
    end
    for (I1, I2) in (
      (mortar([Block(2)[2:3], Block(3)[1:3]]), mortar([Block(2)[2:3], Block(3)[2:3]])),
      ([Block(2)[2:3], Block(3)[1:3]], [Block(2)[2:3], Block(3)[2:3]]),
    )
      for b in (a[I1, I2], @view(a[I1, I2]))
        # TODO: Rename `block_stored_length`.
        @test block_nstored(b) == 2
        @test b[Block(1, 1)] == a[Block(2, 2)[2:3, 2:3]]
        @test b[Block(2, 2)] == a[Block(3, 3)[1:3, 2:3]]
      end
    end

    a = BlockSparseArray{elt}(undef, ([3, 3], [3, 3]))
    # TODO: Define `block_diagindices`.
    @views for b in [Block(1, 1), Block(2, 2)]
      a[b] = randn(elt, size(a[b]))
    end
    I = mortar([Block(1)[1:2], Block(2)[1:2]])
    b = a[:, I]
    @test b[Block(1, 1)] == a[Block(1, 1)][:, 1:2]
    @test b[Block(2, 1)] == a[Block(2, 1)][:, 1:2]
    @test b[Block(1, 2)] == a[Block(1, 2)][:, 1:2]
    @test b[Block(2, 2)] == a[Block(2, 2)][:, 1:2]
    @test blocklengths.(axes(b)) == ([3, 3], [2, 2])
    # TODO: Rename `block_stored_length`.
    @test blocksize(b) == (2, 2)
    @test block_nstored(b) == 2

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    @test isassigned(a, 1, 1)
    @test isassigned(a, 5, 7)
    @test !isassigned(a, 0, 1)
    @test !isassigned(a, 5, 8)
    @test isassigned(a, Block(1), Block(1))
    @test isassigned(a, Block(2), Block(2))
    @test !isassigned(a, Block(1), Block(0))
    @test !isassigned(a, Block(3), Block(2))
    @test isassigned(a, Block(1, 1))
    @test isassigned(a, Block(2, 2))
    @test !isassigned(a, Block(1, 0))
    @test !isassigned(a, Block(3, 2))
    @test isassigned(a, Block(1)[1], Block(1)[1])
    @test isassigned(a, Block(2)[3], Block(2)[4])
    @test !isassigned(a, Block(1)[0], Block(1)[1])
    @test !isassigned(a, Block(2)[3], Block(2)[5])
    @test !isassigned(a, Block(1)[1], Block(0)[1])
    @test !isassigned(a, Block(3)[3], Block(2)[4])

    a = BlockSparseArray{elt}([2, 3], [3, 4])
    @test iszero(a)
    @test iszero(block_nstored(a))
    fill!(a, 0)
    @test iszero(a)
    @test iszero(block_nstored(a))
    fill!(a, 2)
    @test !iszero(a)
    @test all(==(2), a)
    @test block_nstored(a) == 4
    fill!(a, 0)
    @test iszero(a)
    @test iszero(block_nstored(a))

    a = BlockSparseArray{elt}([2, 3], [3, 4])
    @test iszero(a)
    @test iszero(block_nstored(a))
    a .= 0
    @test iszero(a)
    @test iszero(block_nstored(a))
    a .= 2
    @test !iszero(a)
    @test all(==(2), a)
    @test block_nstored(a) == 4
    a .= 0
    @test iszero(a)
    @test iszero(block_nstored(a))

    a = BlockSparseArray{elt}([2, 3], [3, 4])
    for I in (Block.(1:2), [Block(1), Block(2)])
      b = @view a[I, I]
      x = randn(elt, 3, 4)
      b[Block(2, 2)] = x
      # These outputs a block of zeros,
      # for some reason the block
      # is not getting set.
      # I think the issue is that:
      # ```julia
      # @view(@view(a[I, I]))[Block(1, 1)]
      # ```
      # creates a doubly-wrapped SubArray
      # instead of flattening down to a
      # single SubArray wrapper.
      @test a[Block(2, 2)] == x
      @test b[Block(2, 2)] == x
    end

    function f1()
      a = BlockSparseArray{elt}([2, 3], [3, 4])
      b = @view a[[Block(2), Block(1)], [Block(2), Block(1)]]
      x = randn(elt, 3, 4)
      b[Block(1, 1)] .= x
      return (; a, b, x)
    end
    function f2()
      a = BlockSparseArray{elt}([2, 3], [3, 4])
      b = @view a[[Block(2), Block(1)], [Block(2), Block(1)]]
      x = randn(elt, 3, 4)
      b[Block(1, 1)] = x
      return (; a, b, x)
    end
    for abx in (f1(), f2())
      @compat (; a, b, x) = abx
      @test b isa SubArray{<:Any,<:Any,<:BlockSparseArray}
      @test block_nstored(b) == 1
      @test b[Block(1, 1)] == x
      for blck in [Block(2, 1), Block(1, 2), Block(2, 2)]
        @test iszero(b[blck])
      end
      @test block_nstored(a) == 1
      @test a[Block(2, 2)] == x
      for blck in [Block(1, 1), Block(2, 1), Block(1, 2)]
        @test iszero(a[blck])
      end
      @test_throws DimensionMismatch b[Block(1, 1)] .= randn(2, 3)
    end

    a = BlockSparseArray{elt}([2, 3], [3, 4])
    b = @views a[[Block(2), Block(1)], [Block(2), Block(1)]][Block(2, 1)]
    @test iszero(b)
    @test size(b) == (2, 4)
    x = randn(elt, 2, 4)
    b .= x
    @test b == x
    @test a[Block(1, 2)] == x
    @test block_nstored(a) == 1

    a = BlockSparseArray{elt}([2, 3], [3, 4])
    b = @view a[[Block(2), Block(1)], [Block(2), Block(1)]]
    for index in parentindices(@view(b[Block(1, 1)]))
      @test index isa BlockSlice{<:Block{1}}
    end

    a = BlockSparseArray{elt}([2, 3], [3, 4])
    b = @view a[Block(1, 1)[1:2, 1:1]]
    for i in parentindices(b)
      @test i isa BlockSlice{<:BlockIndexRange{1}}
    end
  end
  @testset "view!" begin
    for blk in ((Block(2, 2),), (Block(2), Block(2)))
      a = BlockSparseArray{elt}([2, 3], [2, 3])
      b = view!(a, blk...)
      x = randn(elt, 3, 3)
      b .= x
      @test b == x
      @test a[blk...] == x
      @test @view(a[blk...]) == x
      @test view!(a, blk...) == x
      @test @view!(a[blk...]) == x
    end
    for blk in ((Block(2, 2),), (Block(2), Block(2)))
      a = BlockSparseArray{elt}([2, 3], [2, 3])
      b = @view! a[blk...]
      x = randn(elt, 3, 3)
      b .= x
      @test b == x
      @test a[blk...] == x
      @test @view(a[blk...]) == x
      @test view!(a, blk...) == x
      @test @view!(a[blk...]) == x
    end
    for blk in ((Block(2, 2)[2:3, 1:2],), (Block(2)[2:3], Block(2)[1:2]))
      a = BlockSparseArray{elt}([2, 3], [2, 3])
      b = view!(a, blk...)
      x = randn(elt, 2, 2)
      b .= x
      @test b == x
      @test a[blk...] == x
      @test @view(a[blk...]) == x
      @test view!(a, blk...) == x
      @test @view!(a[blk...]) == x
    end
    for blk in ((Block(2, 2)[2:3, 1:2],), (Block(2)[2:3], Block(2)[1:2]))
      a = BlockSparseArray{elt}([2, 3], [2, 3])
      b = @view! a[blk...]
      x = randn(elt, 2, 2)
      b .= x
      @test b == x
      @test a[blk...] == x
      @test @view(a[blk...]) == x
      @test view!(a, blk...) == x
      @test @view!(a[blk...]) == x
    end
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
  @testset "Matrix multiplication" begin
    a1 = BlockSparseArray{elt}([2, 3], [2, 3])
    a1[Block(1, 2)] = randn(elt, size(@view(a1[Block(1, 2)])))
    a1[Block(2, 1)] = randn(elt, size(@view(a1[Block(2, 1)])))
    a2 = BlockSparseArray{elt}([2, 3], [2, 3])
    a2[Block(1, 2)] = randn(elt, size(@view(a2[Block(1, 2)])))
    a2[Block(2, 1)] = randn(elt, size(@view(a2[Block(2, 1)])))
    @test Array(a1 * a2) ≈ Array(a1) * Array(a2)
    @test Array(a1' * a2) ≈ Array(a1') * Array(a2)
    @test Array(a1 * a2') ≈ Array(a1) * Array(a2')
    @test Array(a1' * a2') ≈ Array(a1') * Array(a2')
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
