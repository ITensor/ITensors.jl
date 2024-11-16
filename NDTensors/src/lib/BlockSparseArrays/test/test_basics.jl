@eval module $(gensym())
using BlockArrays:
  Block,
  BlockIndexRange,
  BlockRange,
  BlockSlice,
  BlockVector,
  BlockedOneTo,
  BlockedUnitRange,
  BlockedVector,
  blockedrange,
  blocklength,
  blocklengths,
  blocksize,
  blocksizes,
  mortar
using Compat: @compat
using GPUArraysCore: @allowscalar
using LinearAlgebra: Adjoint, Transpose, dot, mul!, norm
using NDTensors.BlockSparseArrays:
  @view!,
  BlockSparseArray,
  BlockSparseMatrix,
  BlockSparseVector,
  BlockView,
  block_stored_length,
  block_reshape,
  block_stored_indices,
  blockstype,
  blocktype,
  view!
using NDTensors.GPUArraysCoreExtensions: cpu
using NDTensors.SparseArraysBase: stored_length
using NDTensors.SparseArraysBase: SparseArrayDOK, SparseMatrixDOK, SparseVectorDOK
using NDTensors.TensorAlgebra: contract
using Test: @test, @test_broken, @test_throws, @testset, @inferred
include("TestBlockSparseArraysUtils.jl")

using NDTensors: NDTensors
include(joinpath(pkgdir(NDTensors), "test", "NDTensorsTestUtils", "NDTensorsTestUtils.jl"))
using .NDTensorsTestUtils: devices_list, is_supported_eltype
@testset "BlockSparseArrays (dev=$dev, eltype=$elt)" for dev in devices_list(copy(ARGS)),
  elt in (Float32, Float64, Complex{Float32}, Complex{Float64})

  if !is_supported_eltype(dev, elt)
    continue
  end
  @testset "Broken" begin
    # TODO: Fix this and turn it into a proper test.
    a = dev(BlockSparseArray{elt}([2, 3], [2, 3]))
    a[Block(1, 1)] = dev(randn(elt, 2, 2))
    a[Block(2, 2)] = dev(randn(elt, 3, 3))
    @test_broken a[:, 4]

    # TODO: Fix this and turn it into a proper test.
    a = dev(BlockSparseArray{elt}([2, 3], [2, 3]))
    a[Block(1, 1)] = dev(randn(elt, 2, 2))
    a[Block(2, 2)] = dev(randn(elt, 3, 3))
    @test_broken a[:, [2, 4]]
    @test_broken a[[3, 5], [2, 4]]

    # TODO: Fix this and turn it into a proper test.
    a = dev(BlockSparseArray{elt}([2, 3], [2, 3]))
    a[Block(1, 1)] = dev(randn(elt, 2, 2))
    a[Block(2, 2)] = dev(randn(elt, 3, 3))
    @allowscalar @test a[2:4, 4] == Array(a)[2:4, 4]
    @test_broken a[4, 2:4]

    @test a[Block(1), :] isa BlockSparseArray{elt}
    @test adjoint(a) isa Adjoint{elt,<:BlockSparseArray}
    @test_broken adjoint(a)[Block(1), :] isa Adjoint{elt,<:BlockSparseArray}
    # could also be directly a BlockSparseArray
  end
  @testset "Constructors" begin
    # BlockSparseMatrix
    bs = ([2, 3], [3, 4])
    for T in (
      BlockSparseArray{elt},
      BlockSparseArray{elt,2},
      BlockSparseMatrix{elt},
      BlockSparseArray{elt,2,Matrix{elt}},
      BlockSparseMatrix{elt,Matrix{elt}},
      ## BlockSparseArray{elt,2,Matrix{elt},SparseMatrixDOK{Matrix{elt}}}, # TODO
      ## BlockSparseMatrix{elt,Matrix{elt},SparseMatrixDOK{Matrix{elt}}}, # TODO
    )
      for args in (
        bs,
        (bs,),
        blockedrange.(bs),
        (blockedrange.(bs),),
        (undef, bs),
        (undef, bs...),
        (undef, blockedrange.(bs)),
        (undef, blockedrange.(bs)...),
      )
        a = T(args...)
        @test eltype(a) == elt
        @test blocktype(a) == Matrix{elt}
        @test blockstype(a) <: SparseMatrixDOK{Matrix{elt}}
        @test blocklengths.(axes(a)) == ([2, 3], [3, 4])
        @test iszero(a)
        @test iszero(block_stored_length(a))
        @test iszero(stored_length(a))
      end
    end

    # BlockSparseVector
    bs = ([2, 3],)
    for T in (
      BlockSparseArray{elt},
      BlockSparseArray{elt,1},
      BlockSparseVector{elt},
      BlockSparseArray{elt,1,Vector{elt}},
      BlockSparseVector{elt,Vector{elt}},
      ## BlockSparseArray{elt,1,Vector{elt},SparseVectorDOK{Vector{elt}}}, # TODO
      ## BlockSparseVector{elt,Vector{elt},SparseVectorDOK{Vector{elt}}}, # TODO
    )
      for args in (
        bs,
        (bs,),
        blockedrange.(bs),
        (blockedrange.(bs),),
        (undef, bs),
        (undef, bs...),
        (undef, blockedrange.(bs)),
        (undef, blockedrange.(bs)...),
      )
        a = T(args...)
        @test eltype(a) == elt
        @test blocktype(a) == Vector{elt}
        @test blockstype(a) <: SparseVectorDOK{Vector{elt}}
        @test blocklengths.(axes(a)) == ([2, 3],)
        @test iszero(a)
        @test iszero(block_stored_length(a))
        @test iszero(stored_length(a))
      end
    end
  end
  @testset "Basics" begin
    a = dev(BlockSparseArray{elt}([2, 3], [2, 3]))
    @allowscalar @test a == dev(
      BlockSparseArray{elt}(blockedrange([2, 3]), blockedrange([2, 3]))
    )
    @test eltype(a) === elt
    @test axes(a) == (1:5, 1:5)
    @test all(aᵢ -> aᵢ isa BlockedOneTo, axes(a))
    @test blocklength.(axes(a)) == (2, 2)
    @test blocksize(a) == (2, 2)
    @test size(a) == (5, 5)
    @test block_stored_length(a) == 0
    @test iszero(a)
    @allowscalar @test all(I -> iszero(a[I]), eachindex(a))
    @test_throws DimensionMismatch a[Block(1, 1)] = randn(elt, 2, 3)

    a = BlockSparseArray{elt}([2, 3], [2, 3])
    a[3, 3] = 33
    @test eltype(a) === elt
    @test axes(a) == (1:5, 1:5)
    @test all(aᵢ -> aᵢ isa BlockedOneTo, axes(a))
    @test blocklength.(axes(a)) == (2, 2)
    @test blocksize(a) == (2, 2)
    @test size(a) == (5, 5)
    @test block_stored_length(a) == 1
    @test !iszero(a)
    @test a[3, 3] == 33
    @test all(eachindex(a)) do I
      if I == CartesianIndex(3, 3)
        a[I] == 33
      else
        iszero(a[I])
      end
    end

    a[3, 3] = NaN
    @test isnan(norm(a))

    # Empty constructor
    for a in (dev(BlockSparseArray{elt}()), dev(BlockSparseArray{elt}(undef)))
      @test size(a) == ()
      @test isone(length(a))
      @test blocksize(a) == ()
      @test blocksizes(a) == fill(())
      @test iszero(block_stored_length(a))
      @test iszero(@allowscalar(a[]))
      @test iszero(@allowscalar(a[CartesianIndex()]))
      @test a[Block()] == dev(fill(0))
      @test iszero(@allowscalar(a[Block()][]))
      # Broken:
      ## @test b[Block()[]] == 2
      for b in (
        (b = copy(a); @allowscalar b[] = 2; b),
        (b = copy(a); @allowscalar b[CartesianIndex()] = 2; b),
      )
        @test size(b) == ()
        @test isone(length(b))
        @test blocksize(b) == ()
        @test blocksizes(b) == fill(())
        @test isone(block_stored_length(b))
        @test @allowscalar(b[]) == 2
        @test @allowscalar(b[CartesianIndex()]) == 2
        @test b[Block()] == dev(fill(2))
        @test @allowscalar(b[Block()][]) == 2
        # Broken:
        ## @test b[Block()[]] == 2
      end
    end

    @testset "Transpose" begin
      a = dev(BlockSparseArray{elt}([2, 2], [3, 3, 1]))
      a[Block(1, 1)] = dev(randn(elt, 2, 3))
      a[Block(2, 3)] = dev(randn(elt, 2, 1))

      at = @inferred transpose(a)
      @test at isa Transpose
      @test size(at) == reverse(size(a))
      @test blocksize(at) == reverse(blocksize(a))
      @test stored_length(at) == stored_length(a)
      @test block_stored_length(at) == block_stored_length(a)
      for bind in block_stored_indices(a)
        bindt = Block(reverse(Int.(Tuple(bind))))
        @test bindt in block_stored_indices(at)
      end

      @test @views(at[Block(1, 1)]) == transpose(a[Block(1, 1)])
      @test @views(at[Block(1, 1)]) isa Transpose
      @test @views(at[Block(3, 2)]) == transpose(a[Block(2, 3)])
      # TODO: BlockView == AbstractArray calls scalar code
      @test @allowscalar @views(at[Block(1, 2)]) == transpose(a[Block(2, 1)])
      @test @views(at[Block(1, 2)]) isa Transpose
    end

    @testset "Adjoint" begin
      a = dev(BlockSparseArray{elt}([2, 2], [3, 3, 1]))
      a[Block(1, 1)] = dev(randn(elt, 2, 3))
      a[Block(2, 3)] = dev(randn(elt, 2, 1))

      at = @inferred adjoint(a)
      @test at isa Adjoint
      @test size(at) == reverse(size(a))
      @test blocksize(at) == reverse(blocksize(a))
      @test stored_length(at) == stored_length(a)
      @test block_stored_length(at) == block_stored_length(a)
      for bind in block_stored_indices(a)
        bindt = Block(reverse(Int.(Tuple(bind))))
        @test bindt in block_stored_indices(at)
      end

      @test @views(at[Block(1, 1)]) == adjoint(a[Block(1, 1)])
      @test @views(at[Block(1, 1)]) isa Adjoint
      @test @views(at[Block(3, 2)]) == adjoint(a[Block(2, 3)])
      # TODO: BlockView == AbstractArray calls scalar code
      @test @allowscalar @views(at[Block(1, 2)]) == adjoint(a[Block(2, 1)])
      @test @views(at[Block(1, 2)]) isa Adjoint
    end
  end
  @testset "Tensor algebra" begin
    a = dev(BlockSparseArray{elt}(undef, ([2, 3], [3, 4])))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = dev(randn(elt, size(a[b])))
    end
    @test eltype(a) == elt
    @test block_stored_length(a) == 2
    @test stored_length(a) == 2 * 4 + 3 * 3

    # TODO: Broken on GPU.
    if dev ≠ cpu
      a = dev(BlockSparseArray{elt}([2, 3], [3, 4]))
      @test_broken a[Block(1, 2)] .= 2
    end

    # TODO: Broken on GPU.
    a = BlockSparseArray{elt}([2, 3], [3, 4])
    a[Block(1, 2)] .= 2
    @test eltype(a) == elt
    @test all(==(2), a[Block(1, 2)])
    @test iszero(a[Block(1, 1)])
    @test iszero(a[Block(2, 1)])
    @test iszero(a[Block(2, 2)])
    @test block_stored_length(a) == 1
    @test stored_length(a) == 2 * 4

    # TODO: Broken on GPU.
    if dev ≠ cpu
      a = dev(BlockSparseArray{elt}([2, 3], [3, 4]))
      @test_broken a[Block(1, 2)] .= 0
    end

    # TODO: Broken on GPU.
    a = BlockSparseArray{elt}([2, 3], [3, 4])
    a[Block(1, 2)] .= 0
    @test eltype(a) == elt
    @test iszero(a[Block(1, 1)])
    @test iszero(a[Block(2, 1)])
    @test iszero(a[Block(1, 2)])
    @test iszero(a[Block(2, 2)])
    @test block_stored_length(a) == 1
    @test stored_length(a) == 2 * 4

    a = dev(BlockSparseArray{elt}(undef, ([2, 3], [3, 4])))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = dev(randn(elt, size(a[b])))
    end
    b = similar(a, complex(elt))
    @test eltype(b) == complex(eltype(a))
    @test iszero(b)
    @test block_stored_length(b) == 0
    @test stored_length(b) == 0
    @test size(b) == size(a)
    @test blocksize(b) == blocksize(a)

    a = dev(BlockSparseArray{elt}([2, 3], [3, 4]))
    b = @view a[[Block(2), Block(1)], [Block(2), Block(1)]]
    c = @view b[Block(1, 1)]
    @test iszero(a)
    @test iszero(stored_length(a))
    @test iszero(b)
    @test iszero(stored_length(b))
    # TODO: Broken on GPU.
    @test iszero(c) broken = dev ≠ cpu
    @test iszero(stored_length(c))
    @allowscalar a[5, 7] = 1
    @test !iszero(a)
    @test stored_length(a) == 3 * 4
    @test !iszero(b)
    @test stored_length(b) == 3 * 4
    # TODO: Broken on GPU.
    @test !iszero(c) broken = dev ≠ cpu
    @test stored_length(c) == 3 * 4
    d = @view a[1:4, 1:6]
    @test iszero(d)
    @test stored_length(d) == 2 * 3

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

    a = dev(BlockSparseArray{elt}(undef, ([2, 3], [3, 4])))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = dev(randn(elt, size(a[b])))
    end
    b = 2 * a
    @allowscalar @test Array(b) ≈ 2 * Array(a)
    @test eltype(b) == elt
    @test block_stored_length(b) == 2
    @test stored_length(b) == 2 * 4 + 3 * 3

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = (2 + 3im) * a
    @test Array(b) ≈ (2 + 3im) * Array(a)
    @test eltype(b) == complex(elt)
    @test block_stored_length(b) == 2
    @test stored_length(b) == 2 * 4 + 3 * 3

    a = dev(BlockSparseArray{elt}(undef, ([2, 3], [3, 4])))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = dev(randn(elt, size(a[b])))
    end
    b = a + a
    @allowscalar @test Array(b) ≈ 2 * Array(a)
    @test eltype(b) == elt
    @test block_stored_length(b) == 2
    @test stored_length(b) == 2 * 4 + 3 * 3

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
    @test block_stored_length(b) == 2
    @test stored_length(b) == 2 * 4 + 3 * 3

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = permutedims(a, (2, 1))
    @test Array(b) ≈ permutedims(Array(a), (2, 1))
    @test eltype(b) == elt
    @test block_stored_length(b) == 2
    @test stored_length(b) == 2 * 4 + 3 * 3

    a = dev(BlockSparseArray{elt}([1, 1, 1], [1, 2, 3], [2, 2, 1], [1, 2, 1]))
    a[Block(3, 2, 2, 3)] = dev(randn(elt, 1, 2, 2, 1))
    perm = (2, 3, 4, 1)
    for b in (PermutedDimsArray(a, perm), permutedims(a, perm))
      @test Array(b) == permutedims(Array(a), perm)
      @test issetequal(block_stored_indices(b), [Block(2, 2, 3, 3)])
      @test @allowscalar b[Block(2, 2, 3, 3)] == permutedims(a[Block(3, 2, 2, 3)], perm)
    end

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = map(x -> 2x, a)
    @test Array(b) ≈ 2 * Array(a)
    @test eltype(b) == elt
    @test size(b) == size(a)
    @test blocksize(b) == (2, 2)
    @test block_stored_length(b) == 2
    @test stored_length(b) == 2 * 4 + 3 * 3

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
    @test stored_length(b) == stored_length(a)
    @test block_stored_length(b) == 2

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = a[Block(1):Block(2), Block(1):Block(2)]
    @test b == a
    @test size(b) == size(a)
    @test blocksize(b) == (2, 2)
    @test stored_length(b) == stored_length(a)
    @test block_stored_length(b) == 2

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
    @test stored_length(b) == stored_length(a[Block(1, 2)])
    @test block_stored_length(b) == 1

    a = dev(BlockSparseArray{elt}(undef, ([2, 3], [3, 4])))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = dev(randn(elt, size(a[b])))
    end
    for b in (a[2:4, 2:4], @view(a[2:4, 2:4]))
      @allowscalar @test b == Array(a)[2:4, 2:4]
      @test size(b) == (3, 3)
      @test blocksize(b) == (2, 2)
      @test stored_length(b) == 1 * 1 + 2 * 2
      @test block_stored_length(b) == 2
      for f in (getindex, view)
        # TODO: Broken on GPU.
        @allowscalar begin
          @test size(f(b, Block(1, 1))) == (1, 2)
          @test size(f(b, Block(2, 1))) == (2, 2)
          @test size(f(b, Block(1, 2))) == (1, 1)
          @test size(f(b, Block(2, 2))) == (2, 1)
          @test f(b, Block(1, 1)) == a[Block(1, 1)[2:2, 2:3]]
          @test f(b, Block(2, 1)) == a[Block(2, 1)[1:2, 2:3]]
          @test f(b, Block(1, 2)) == a[Block(1, 2)[2:2, 1:1]]
          @test f(b, Block(2, 2)) == a[Block(2, 2)[1:2, 1:1]]
        end
      end
    end

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = a[Block(2, 1)[1:2, 2:3]]
    @test b == Array(a)[3:4, 2:3]
    @test size(b) == (2, 2)
    @test blocksize(b) == (1, 1)
    @test stored_length(b) == 2 * 2
    @test block_stored_length(b) == 1

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = PermutedDimsArray(a, (2, 1))
    @test block_stored_length(b) == 2
    @test Array(b) == permutedims(Array(a), (2, 1))
    c = 2 * b
    @test block_stored_length(c) == 2
    @test Array(c) == 2 * permutedims(Array(a), (2, 1))

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = a'
    @test block_stored_length(b) == 2
    @test Array(b) == Array(a)'
    c = 2 * b
    @test block_stored_length(c) == 2
    @test Array(c) == 2 * Array(a)'

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = transpose(a)
    @test block_stored_length(b) == 2
    @test Array(b) == transpose(Array(a))
    c = 2 * b
    @test block_stored_length(c) == 2
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
      @test i isa Base.OneTo{Int}
    end
    @test parentindices(b)[1] == 1:3
    @test parentindices(b)[2] == 1:4

    a = BlockSparseArray{elt}([2, 3], [3, 4])
    b = @view a[Block(2, 2)[1:2, 2:2]]
    @test size(b) == (2, 1)
    for i in parentindices(b)
      @test i isa UnitRange{Int}
    end
    @test parentindices(b)[1] == 1:2
    @test parentindices(b)[2] == 2:2

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

    a = BlockSparseArray{elt}([2, 3], [3, 4])
    b = @views a[Block(2, 2)][1:2, 2:3]
    @test b isa SubArray{<:Any,<:Any,<:BlockView}
    for i in parentindices(b)
      @test i isa UnitRange{Int}
    end
    x = randn(elt, 2, 2)
    b .= x
    @test a[Block(2, 2)[1:2, 2:3]] == x
    @test a[Block(2, 2)[1:2, 2:3]] == b
    @test block_stored_length(a) == 1

    a = BlockSparseArray{elt}([2, 3], [2, 3])
    @views for b in [Block(1, 1), Block(2, 2)]
      a[b] = randn(elt, size(a[b]))
    end
    for I in (Block.(1:2), [Block(1), Block(2)])
      b = @view a[I, I]
      for I in CartesianIndices(a)
        @test b[I] == a[I]
      end
      for block in BlockRange(a)
        @test b[block] == a[block]
      end
    end

    a = dev(BlockSparseArray{elt}([2, 3], [2, 3]))
    @views for b in [Block(1, 1), Block(2, 2)]
      # TODO: Use `blocksizes(a)[Int.(Tuple(b))...]` once available.
      a[b] = dev(randn(elt, size(a[b])))
    end
    for I in ([Block(2), Block(1)],)
      b = @view a[I, I]
      @test b[Block(1, 1)] == a[Block(2, 2)]
      @test b[Block(2, 1)] == a[Block(1, 2)]
      @test b[Block(1, 2)] == a[Block(2, 1)]
      @test b[Block(2, 2)] == a[Block(1, 1)]
      @allowscalar begin
        @test b[1, 1] == a[3, 3]
        @test b[4, 4] == a[1, 1]
        b[4, 4] = 44
        @test b[4, 4] == 44
      end
    end

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    b = a[Block(2):Block(2), Block(1):Block(2)]
    @test block_stored_length(b) == 1
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
        @test block_stored_length(b) == 2
        @test b[Block(1, 1)] == a[Block(2, 2)[2:3, 2:3]]
        @test b[Block(2, 2)] == a[Block(3, 3)[1:3, 2:3]]
      end
    end

    a = dev(BlockSparseArray{elt}(undef, ([3, 3], [3, 3])))
    # TODO: Define `block_diagindices`.
    @views for b in [Block(1, 1), Block(2, 2)]
      a[b] = dev(randn(elt, size(a[b])))
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
    @test block_stored_length(b) == 2

    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    @views for b in [Block(1, 2), Block(2, 1)]
      a[b] = randn(elt, size(a[b]))
    end
    @test isassigned(a, 1, 1)
    @test isassigned(a, 1, 1, 1)
    @test !isassigned(a, 1, 1, 2)
    @test isassigned(a, 5, 7)
    @test isassigned(a, 5, 7, 1)
    @test !isassigned(a, 5, 7, 2)
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
    @test iszero(block_stored_length(a))
    fill!(a, 0)
    @test iszero(a)
    @test iszero(block_stored_length(a))
    fill!(a, 2)
    @test !iszero(a)
    @test all(==(2), a)
    @test block_stored_length(a) == 4
    fill!(a, 0)
    @test iszero(a)
    @test iszero(block_stored_length(a))

    a = BlockSparseArray{elt}([2, 3], [3, 4])
    @test iszero(a)
    @test iszero(block_stored_length(a))
    a .= 0
    @test iszero(a)
    @test iszero(block_stored_length(a))
    a .= 2
    @test !iszero(a)
    @test all(==(2), a)
    @test block_stored_length(a) == 4
    a .= 0
    @test iszero(a)
    @test iszero(block_stored_length(a))

    # TODO: Broken on GPU.
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
      @test block_stored_length(b) == 1
      @test b[Block(1, 1)] == x
      @test @view(b[Block(1, 1)]) isa Matrix{elt}
      for blck in [Block(2, 1), Block(1, 2), Block(2, 2)]
        @test iszero(b[blck])
      end
      @test block_stored_length(a) == 1
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
    @test block_stored_length(a) == 1

    a = BlockSparseArray{elt}([4, 3, 2], [4, 3, 2])
    @views for B in [Block(1, 1), Block(2, 2), Block(3, 3)]
      a[B] = randn(elt, size(a[B]))
    end
    b = @view a[[Block(3), Block(2), Block(1)], [Block(3), Block(2), Block(1)]]
    @test b isa SubArray{<:Any,<:Any,<:BlockSparseArray}
    c = @view b[4:8, 4:8]
    @test c isa SubArray{<:Any,<:Any,<:BlockSparseArray}
    @test size(c) == (5, 5)
    @test block_stored_length(c) == 2
    @test blocksize(c) == (2, 2)
    @test blocklengths.(axes(c)) == ([2, 3], [2, 3])
    @test size(c[Block(1, 1)]) == (2, 2)
    @test c[Block(1, 1)] == a[Block(2, 2)[2:3, 2:3]]
    @test size(c[Block(2, 2)]) == (3, 3)
    @test c[Block(2, 2)] == a[Block(1, 1)[1:3, 1:3]]
    @test size(c[Block(2, 1)]) == (3, 2)
    @test iszero(c[Block(2, 1)])
    @test size(c[Block(1, 2)]) == (2, 3)
    @test iszero(c[Block(1, 2)])

    x = randn(elt, 3, 3)
    c[Block(2, 2)] = x
    @test c[Block(2, 2)] == x
    @test a[Block(1, 1)[1:3, 1:3]] == x

    a = BlockSparseArray{elt}([2, 3], [3, 4])
    b = @view a[[Block(2), Block(1)], [Block(2), Block(1)]]
    for index in parentindices(@view(b[Block(1, 1)]))
      @test index isa Base.OneTo{Int}
    end

    a = BlockSparseArray{elt}([2, 3], [3, 4])
    a[Block(1, 1)] = randn(elt, 2, 3)
    b = @view a[Block(1, 1)[1:2, 1:1]]
    @test b isa SubArray{elt,2,Matrix{elt}}
    for i in parentindices(b)
      @test i isa UnitRange{Int}
    end

    a = BlockSparseArray{elt}([2, 2, 2, 2], [2, 2, 2, 2])
    @views for I in [Block(1, 1), Block(2, 2), Block(3, 3), Block(4, 4)]
      a[I] = randn(elt, size(a[I]))
    end
    for I in (blockedrange([4, 4]), BlockedVector(Block.(1:4), [2, 2]))
      b = @view a[I, I]
      @test copy(b) == a
      @test blocksize(b) == (2, 2)
      @test blocklengths.(axes(b)) == ([4, 4], [4, 4])
      # TODO: Fix in Julia 1.11 (https://github.com/ITensor/ITensors.jl/pull/1539).
      if VERSION < v"1.11-"
        @test b[Block(1, 1)] == a[Block.(1:2), Block.(1:2)]
        @test b[Block(2, 1)] == a[Block.(3:4), Block.(1:2)]
        @test b[Block(1, 2)] == a[Block.(1:2), Block.(3:4)]
        @test b[Block(2, 2)] == a[Block.(3:4), Block.(3:4)]
      end
      c = @view b[Block(2, 2)]
      @test blocksize(c) == (1, 1)
      @test c == a[Block.(3:4), Block.(3:4)]
    end

    a = BlockSparseArray{elt}([2, 3], [2, 3])
    a[Block(1, 1)] = randn(elt, 2, 2)
    a[Block(2, 2)] = randn(elt, 3, 3)
    for I in (mortar([Block(1)[2:2], Block(2)[2:3]]), [Block(1)[2:2], Block(2)[2:3]])
      b = @view a[:, I]
      @test b == Array(a)[:, [2, 4, 5]]
    end

    # Merge and permute blocks.
    a = BlockSparseArray{elt}([2, 2, 2, 2], [2, 2, 2, 2])
    @views for I in [Block(1, 1), Block(2, 2), Block(3, 3), Block(4, 4)]
      a[I] = randn(elt, size(a[I]))
    end
    for I in (
      BlockVector([Block(4), Block(3), Block(2), Block(1)], [2, 2]),
      BlockedVector([Block(4), Block(3), Block(2), Block(1)], [2, 2]),
    )
      b = @view a[I, I]
      J = [Block(4), Block(3), Block(2), Block(1)]
      @test b == a[J, J]
      @test copy(b) == a[J, J]
      @test blocksize(b) == (2, 2)
      @test blocklengths.(axes(b)) == ([4, 4], [4, 4])
      @test b[Block(1, 1)] == Array(a)[[7, 8, 5, 6], [7, 8, 5, 6]]
      c = @views b[Block(1, 1)][2:3, 2:3]
      @test c == Array(a)[[8, 5], [8, 5]]
      @test copy(c) == Array(a)[[8, 5], [8, 5]]
      c = @view b[Block(1, 1)[2:3, 2:3]]
      @test c == Array(a)[[8, 5], [8, 5]]
      @test copy(c) == Array(a)[[8, 5], [8, 5]]
    end

    # TODO: Add more tests of this, it may
    # only be working accidentally.
    a = BlockSparseArray{elt}([2, 3], [2, 3])
    a[Block(1, 1)] = randn(elt, 2, 2)
    a[Block(2, 2)] = randn(elt, 3, 3)
    @test a[2:4, 4] == Array(a)[2:4, 4]
    # TODO: Fix this.
    @test_broken a[4, 2:4] == Array(a)[4, 2:4]
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
    a1 = dev(BlockSparseArray{elt}([2, 3], [2, 3]))
    a1[Block(1, 1)] = dev(randn(elt, size(@view(a1[Block(1, 1)]))))
    a2 = dev(BlockSparseArray{elt}([2, 3], [2, 3]))
    a2[Block(1, 1)] = dev(randn(elt, size(@view(a1[Block(1, 1)]))))
    a_dest = a1 * a2
    @allowscalar @test Array(a_dest) ≈ Array(a1) * Array(a2)
    @test a_dest isa BlockSparseArray{elt}
    @test block_stored_length(a_dest) == 1
  end
  @testset "Matrix multiplication" begin
    a1 = dev(BlockSparseArray{elt}([2, 3], [2, 3]))
    a1[Block(1, 2)] = dev(randn(elt, size(@view(a1[Block(1, 2)]))))
    a1[Block(2, 1)] = dev(randn(elt, size(@view(a1[Block(2, 1)]))))
    a2 = dev(BlockSparseArray{elt}([2, 3], [2, 3]))
    a2[Block(1, 2)] = dev(randn(elt, size(@view(a2[Block(1, 2)]))))
    a2[Block(2, 1)] = dev(randn(elt, size(@view(a2[Block(2, 1)]))))
    for (a1′, a2′) in ((a1, a2), (a1', a2), (a1, a2'), (a1', a2'))
      a_dest = a1′ * a2′
      @allowscalar @test Array(a_dest) ≈ Array(a1′) * Array(a2′)
    end
  end
  @testset "Dot product" begin
    a1 = dev(BlockSparseArray{elt}([2, 3, 4]))
    a1[Block(1)] = dev(randn(elt, size(@view(a1[Block(1)]))))
    a1[Block(3)] = dev(randn(elt, size(@view(a1[Block(3)]))))
    a2 = dev(BlockSparseArray{elt}([2, 3, 4]))
    a2[Block(2)] = dev(randn(elt, size(@view(a1[Block(2)]))))
    a2[Block(3)] = dev(randn(elt, size(@view(a1[Block(3)]))))
    @test a1' * a2 ≈ Array(a1)' * Array(a2)
    @test dot(a1, a2) ≈ a1' * a2
  end
  @testset "cat" begin
    a1 = dev(BlockSparseArray{elt}([2, 3], [2, 3]))
    a1[Block(2, 1)] = dev(randn(elt, size(@view(a1[Block(2, 1)]))))
    a2 = dev(BlockSparseArray{elt}([2, 3], [2, 3]))
    a2[Block(1, 2)] = dev(randn(elt, size(@view(a2[Block(1, 2)]))))

    a_dest = cat(a1, a2; dims=1)
    @test block_stored_length(a_dest) == 2
    @test blocklengths.(axes(a_dest)) == ([2, 3, 2, 3], [2, 3])
    @test issetequal(block_stored_indices(a_dest), [Block(2, 1), Block(3, 2)])
    @test a_dest[Block(2, 1)] == a1[Block(2, 1)]
    @test a_dest[Block(3, 2)] == a2[Block(1, 2)]

    a_dest = cat(a1, a2; dims=2)
    @test block_stored_length(a_dest) == 2
    @test blocklengths.(axes(a_dest)) == ([2, 3], [2, 3, 2, 3])
    @test issetequal(block_stored_indices(a_dest), [Block(2, 1), Block(1, 4)])
    @test a_dest[Block(2, 1)] == a1[Block(2, 1)]
    @test a_dest[Block(1, 4)] == a2[Block(1, 2)]

    a_dest = cat(a1, a2; dims=(1, 2))
    @test block_stored_length(a_dest) == 2
    @test blocklengths.(axes(a_dest)) == ([2, 3, 2, 3], [2, 3, 2, 3])
    @test issetequal(block_stored_indices(a_dest), [Block(2, 1), Block(3, 4)])
    @test a_dest[Block(2, 1)] == a1[Block(2, 1)]
    @test a_dest[Block(3, 4)] == a2[Block(1, 2)]
  end
  @testset "TensorAlgebra" begin
    a1 = dev(BlockSparseArray{elt}([2, 3], [2, 3]))
    a1[Block(1, 1)] = dev(randn(elt, size(@view(a1[Block(1, 1)]))))
    a2 = dev(BlockSparseArray{elt}([2, 3], [2, 3]))
    a2[Block(1, 1)] = dev(randn(elt, size(@view(a1[Block(1, 1)]))))
    # TODO: Make this work, requires customization of `TensorAlgebra.fusedims` and
    # `TensorAlgebra.splitdims` in terms of `BlockSparseArrays.block_reshape`,
    # and customization of `TensorAlgebra.:⊗` in terms of `GradedAxes.tensor_product`.
    a_dest, dimnames_dest = contract(a1, (1, -1), a2, (-1, 2))
    @allowscalar begin
      a_dest_dense, dimnames_dest_dense = contract(Array(a1), (1, -1), Array(a2), (-1, 2))
      @test a_dest ≈ a_dest_dense
    end
  end
  @testset "block_reshape" begin
    a = dev(BlockSparseArray{elt}(undef, ([3, 4], [2, 3])))
    a[Block(1, 2)] = dev(randn(elt, size(@view(a[Block(1, 2)]))))
    a[Block(2, 1)] = dev(randn(elt, size(@view(a[Block(2, 1)]))))
    b = block_reshape(a, [6, 8, 9, 12])
    @test reshape(a[Block(1, 2)], 9) == b[Block(3)]
    @test reshape(a[Block(2, 1)], 8) == b[Block(2)]
    @test block_stored_length(b) == 2
    @test stored_length(b) == 17
  end
end
end
