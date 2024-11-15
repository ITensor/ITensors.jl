@eval module $(gensym())
using Test: @test, @testset
using BlockArrays:
  AbstractBlockArray, Block, BlockedOneTo, blockedrange, blocklengths, blocksize
using NDTensors.BlockSparseArrays: BlockSparseArray, block_stored_length
using NDTensors.GradedAxes:
  GradedAxes,
  GradedOneTo,
  GradedUnitRange,
  GradedUnitRangeDual,
  blocklabels,
  dual,
  gradedrange,
  isdual
using NDTensors.LabelledNumbers: label
using NDTensors.SparseArraysBase: stored_length
using NDTensors.SymmetrySectors: U1
using NDTensors.TensorAlgebra: fusedims, splitdims
using LinearAlgebra: adjoint
using Random: randn!
function blockdiagonal!(f, a::AbstractArray)
  for i in 1:minimum(blocksize(a))
    b = Block(ntuple(Returns(i), ndims(a)))
    a[b] = f(a[b])
  end
  return a
end

const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "BlockSparseArraysGradedAxesExt (eltype=$elt)" for elt in elts
  @testset "map" begin
    d1 = gradedrange([U1(0) => 2, U1(1) => 2])
    d2 = gradedrange([U1(0) => 2, U1(1) => 2])
    a = BlockSparseArray{elt}(d1, d2, d1, d2)
    blockdiagonal!(randn!, a)
    @test axes(a, 1) isa GradedOneTo
    @test axes(view(a, 1:4, 1:4, 1:4, 1:4), 1) isa GradedOneTo

    for b in (a + a, 2 * a)
      @test size(b) == (4, 4, 4, 4)
      @test blocksize(b) == (2, 2, 2, 2)
      @test blocklengths.(axes(b)) == ([2, 2], [2, 2], [2, 2], [2, 2])
      @test stored_length(b) == 32
      @test block_stored_length(b) == 2
      for i in 1:ndims(a)
        @test axes(b, i) isa GradedOneTo
      end
      @test label(axes(b, 1)[Block(1)]) == U1(0)
      @test label(axes(b, 1)[Block(2)]) == U1(1)
      @test Array(b) isa Array{elt}
      @test Array(b) == b
      @test 2 * Array(a) == b
    end

    # Test mixing graded axes and dense axes
    # in addition/broadcasting.
    for b in (a + Array(a), Array(a) + a)
      @test size(b) == (4, 4, 4, 4)
      @test blocksize(b) == (2, 2, 2, 2)
      @test blocklengths.(axes(b)) == ([2, 2], [2, 2], [2, 2], [2, 2])
      @test stored_length(b) == 256
      @test block_stored_length(b) == 16
      for i in 1:ndims(a)
        @test axes(b, i) isa BlockedOneTo{Int}
      end
      @test Array(a) isa Array{elt}
      @test Array(a) == a
      @test 2 * Array(a) == b
    end

    b = a[2:3, 2:3, 2:3, 2:3]
    @test size(b) == (2, 2, 2, 2)
    @test blocksize(b) == (2, 2, 2, 2)
    @test stored_length(b) == 2
    @test block_stored_length(b) == 2
    for i in 1:ndims(a)
      @test axes(b, i) isa GradedOneTo
    end
    @test label(axes(b, 1)[Block(1)]) == U1(0)
    @test label(axes(b, 1)[Block(2)]) == U1(1)
    @test Array(a) isa Array{elt}
    @test Array(a) == a
  end
  # TODO: Add tests for various slicing operations.
  @testset "fusedims" begin
    d1 = gradedrange([U1(0) => 1, U1(1) => 1])
    d2 = gradedrange([U1(0) => 1, U1(1) => 1])
    a = BlockSparseArray{elt}(d1, d2, d1, d2)
    blockdiagonal!(randn!, a)
    m = fusedims(a, (1, 2), (3, 4))
    for ax in axes(m)
      @test ax isa GradedOneTo
      @test blocklabels(ax) == [U1(0), U1(1), U1(2)]
    end
    for I in CartesianIndices(m)
      if I ∈ CartesianIndex.([(1, 1), (4, 4)])
        @test !iszero(m[I])
      else
        @test iszero(m[I])
      end
    end
    @test a[1, 1, 1, 1] == m[1, 1]
    @test a[2, 2, 2, 2] == m[4, 4]
    @test blocksize(m) == (3, 3)
    @test a == splitdims(m, (d1, d2), (d1, d2))
  end

  @testset "dual axes" begin
    r = gradedrange([U1(0) => 2, U1(1) => 2])
    for ax in ((r, r), (dual(r), r), (r, dual(r)), (dual(r), dual(r)))
      a = BlockSparseArray{elt}(ax...)
      @views for b in [Block(1, 1), Block(2, 2)]
        a[b] = randn(elt, size(a[b]))
      end
      for dim in 1:ndims(a)
        @test typeof(ax[dim]) === typeof(axes(a, dim))
        @test isdual(ax[dim]) == isdual(axes(a, dim))
      end
      @test @view(a[Block(1, 1)])[1, 1] == a[1, 1]
      @test @view(a[Block(1, 1)])[2, 1] == a[2, 1]
      @test @view(a[Block(1, 1)])[1, 2] == a[1, 2]
      @test @view(a[Block(1, 1)])[2, 2] == a[2, 2]
      @test @view(a[Block(2, 2)])[1, 1] == a[3, 3]
      @test @view(a[Block(2, 2)])[2, 1] == a[4, 3]
      @test @view(a[Block(2, 2)])[1, 2] == a[3, 4]
      @test @view(a[Block(2, 2)])[2, 2] == a[4, 4]
      @test @view(a[Block(1, 1)])[1:2, 1:2] == a[1:2, 1:2]
      @test @view(a[Block(2, 2)])[1:2, 1:2] == a[3:4, 3:4]
      a_dense = Array(a)
      @test eachindex(a) == CartesianIndices(size(a))
      for I in eachindex(a)
        @test a[I] == a_dense[I]
      end
      @test axes(a') == dual.(reverse(axes(a)))

      @test isdual(axes(a', 1)) ≠ isdual(axes(a, 2))
      @test isdual(axes(a', 2)) ≠ isdual(axes(a, 1))
      @test isnothing(show(devnull, MIME("text/plain"), a))

      # Check preserving dual in tensor algebra.
      for b in (a + a, 2 * a, 3 * a - a)
        @test Array(b) ≈ 2 * Array(a)
        for dim in 1:ndims(a)
          @test isdual(axes(b, dim)) == isdual(axes(a, dim))
        end
      end

      @test isnothing(show(devnull, MIME("text/plain"), @view(a[Block(1, 1)])))
      @test @view(a[Block(1, 1)]) == a[Block(1, 1)]
    end

    @testset "GradedOneTo" begin
      r = gradedrange([U1(0) => 2, U1(1) => 2])
      a = BlockSparseArray{elt}(r, r)
      @views for i in [Block(1, 1), Block(2, 2)]
        a[i] = randn(elt, size(a[i]))
      end
      b = 2 * a
      @test block_stored_length(b) == 2
      @test Array(b) == 2 * Array(a)
      for i in 1:2
        @test axes(b, i) isa GradedOneTo
        @test axes(a[:, :], i) isa GradedOneTo
      end

      I = [Block(1)[1:1]]
      @test a[I, :] isa AbstractBlockArray
      @test a[:, I] isa AbstractBlockArray
      @test size(a[I, I]) == (1, 1)
      @test !isdual(axes(a[I, I], 1))
    end

    @testset "GradedUnitRange" begin
      r = gradedrange([U1(0) => 2, U1(1) => 2])[1:3]
      a = BlockSparseArray{elt}(r, r)
      @views for i in [Block(1, 1), Block(2, 2)]
        a[i] = randn(elt, size(a[i]))
      end
      b = 2 * a
      @test block_stored_length(b) == 2
      @test Array(b) == 2 * Array(a)
      for i in 1:2
        @test axes(b, i) isa GradedUnitRange
        @test axes(a[:, :], i) isa GradedUnitRange
      end

      I = [Block(1)[1:1]]
      @test a[I, :] isa AbstractBlockArray
      @test axes(a[I, :], 1) isa GradedOneTo
      @test axes(a[I, :], 2) isa GradedUnitRange

      @test a[:, I] isa AbstractBlockArray
      @test axes(a[:, I], 2) isa GradedOneTo
      @test axes(a[:, I], 1) isa GradedUnitRange
      @test size(a[I, I]) == (1, 1)
      @test !isdual(axes(a[I, I], 1))
    end

    # Test case when all axes are dual.
    @testset "dual GradedOneTo" begin
      r = gradedrange([U1(-1) => 2, U1(1) => 2])
      a = BlockSparseArray{elt}(dual(r), dual(r))
      @views for i in [Block(1, 1), Block(2, 2)]
        a[i] = randn(elt, size(a[i]))
      end
      b = 2 * a
      @test block_stored_length(b) == 2
      @test Array(b) == 2 * Array(a)
      for i in 1:2
        @test axes(b, i) isa GradedUnitRangeDual
        @test axes(a[:, :], i) isa GradedUnitRangeDual
      end
      I = [Block(1)[1:1]]
      @test a[I, :] isa AbstractBlockArray
      @test a[:, I] isa AbstractBlockArray
      @test size(a[I, I]) == (1, 1)
      @test isdual(axes(a[I, :], 2))
      @test isdual(axes(a[:, I], 1))
      @test isdual(axes(a[I, :], 1))
      @test isdual(axes(a[:, I], 2))
      @test isdual(axes(a[I, I], 1))
      @test isdual(axes(a[I, I], 2))
    end

    @testset "dual GradedUnitRange" begin
      r = gradedrange([U1(0) => 2, U1(1) => 2])[1:3]
      a = BlockSparseArray{elt}(dual(r), dual(r))
      @views for i in [Block(1, 1), Block(2, 2)]
        a[i] = randn(elt, size(a[i]))
      end
      b = 2 * a
      @test block_stored_length(b) == 2
      @test Array(b) == 2 * Array(a)
      for i in 1:2
        @test axes(b, i) isa GradedUnitRangeDual
        @test axes(a[:, :], i) isa GradedUnitRangeDual
      end

      I = [Block(1)[1:1]]
      @test a[I, :] isa AbstractBlockArray
      @test a[:, I] isa AbstractBlockArray
      @test size(a[I, I]) == (1, 1)
      @test isdual(axes(a[I, :], 2))
      @test isdual(axes(a[:, I], 1))
      @test isdual(axes(a[I, :], 1))
      @test isdual(axes(a[:, I], 2))
      @test isdual(axes(a[I, I], 1))
      @test isdual(axes(a[I, I], 2))
    end

    @testset "dual BlockedUnitRange" begin    # self dual
      r = blockedrange([2, 2])
      a = BlockSparseArray{elt}(dual(r), dual(r))
      @views for i in [Block(1, 1), Block(2, 2)]
        a[i] = randn(elt, size(a[i]))
      end
      b = 2 * a
      @test block_stored_length(b) == 2
      @test Array(b) == 2 * Array(a)
      @test a[:, :] isa BlockSparseArray
      for i in 1:2
        @test axes(b, i) isa BlockedOneTo
        @test axes(a[:, :], i) isa BlockedOneTo
      end

      I = [Block(1)[1:1]]
      @test a[I, :] isa BlockSparseArray
      @test a[:, I] isa BlockSparseArray
      @test size(a[I, I]) == (1, 1)
      @test !isdual(axes(a[I, I], 1))
    end

    # Test case when all axes are dual from taking the adjoint.
    for r in (
      gradedrange([U1(0) => 2, U1(1) => 2]),
      gradedrange([U1(0) => 2, U1(1) => 2])[begin:end],
    )
      a = BlockSparseArray{elt}(r, r)
      @views for i in [Block(1, 1), Block(2, 2)]
        a[i] = randn(elt, size(a[i]))
      end
      b = 2 * a'
      @test block_stored_length(b) == 2
      @test Array(b) == 2 * Array(a)'
      for ax in axes(b)
        @test ax isa typeof(dual(r))
      end

      @test !isdual(axes(a, 1))
      @test !isdual(axes(a, 2))
      @test isdual(axes(a', 1))
      @test isdual(axes(a', 2))
      @test isdual(axes(b, 1))
      @test isdual(axes(b, 2))
      @test isdual(axes(copy(a'), 1))
      @test isdual(axes(copy(a'), 2))

      I = [Block(1)[1:1]]
      @test size(b[I, :]) == (1, 4)
      @test size(b[:, I]) == (4, 1)
      @test size(b[I, I]) == (1, 1)
    end
  end
  @testset "Matrix multiplication" begin
    r = gradedrange([U1(0) => 2, U1(1) => 3])
    a1 = BlockSparseArray{elt}(dual(r), r)
    a1[Block(1, 2)] = randn(elt, size(@view(a1[Block(1, 2)])))
    a1[Block(2, 1)] = randn(elt, size(@view(a1[Block(2, 1)])))
    a2 = BlockSparseArray{elt}(dual(r), r)
    a2[Block(1, 2)] = randn(elt, size(@view(a2[Block(1, 2)])))
    a2[Block(2, 1)] = randn(elt, size(@view(a2[Block(2, 1)])))
    @test Array(a1 * a2) ≈ Array(a1) * Array(a2)
    @test Array(a1' * a2') ≈ Array(a1') * Array(a2')

    a2 = BlockSparseArray{elt}(r, dual(r))
    a2[Block(1, 2)] = randn(elt, size(@view(a2[Block(1, 2)])))
    a2[Block(2, 1)] = randn(elt, size(@view(a2[Block(2, 1)])))
    @test Array(a1' * a2) ≈ Array(a1') * Array(a2)
    @test Array(a1 * a2') ≈ Array(a1) * Array(a2')
  end
end
end
