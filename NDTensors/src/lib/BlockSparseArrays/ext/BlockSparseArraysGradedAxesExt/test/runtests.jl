@eval module $(gensym())
using Compat: Returns
using Test: @test, @testset, @test_broken
using BlockArrays: Block, blockedrange, blocksize
using NDTensors.BlockSparseArrays: BlockSparseArray, block_nstored
using NDTensors.GradedAxes:
  GradedAxes, GradedUnitRange, UnitRangeDual, blocklabels, dual, gradedrange
using NDTensors.LabelledNumbers: label
using NDTensors.SparseArrayInterface: nstored
using NDTensors.TensorAlgebra: fusedims, splitdims
using Random: randn!
function blockdiagonal!(f, a::AbstractArray)
  for i in 1:minimum(blocksize(a))
    b = Block(ntuple(Returns(i), ndims(a)))
    a[b] = f(a[b])
  end
  return a
end

struct U1
  n::Int
end
GradedAxes.dual(c::U1) = U1(-c.n)
GradedAxes.fuse_labels(c1::U1, c2::U1) = U1(c1.n + c2.n)
Base.isless(c1::U1, c2::U1) = isless(c1.n, c2.n)

const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "BlockSparseArraysGradedAxesExt (eltype=$elt)" for elt in elts
  @testset "map" begin
    d1 = gradedrange([U1(0) => 2, U1(1) => 2])
    d2 = gradedrange([U1(0) => 2, U1(1) => 2])
    a = BlockSparseArray{elt}(d1, d2, d1, d2)
    blockdiagonal!(randn!, a)

    for b in (a + a, 2 * a)
      @test size(b) == (4, 4, 4, 4)
      @test blocksize(b) == (2, 2, 2, 2)
      @test nstored(b) == 32
      @test block_nstored(b) == 2
      # TODO: Have to investigate why this fails
      # on Julia v1.6, or drop support for v1.6.
      for i in 1:ndims(a)
        @test axes(b, i) isa GradedUnitRange
      end
      @test label(axes(b, 1)[Block(1)]) == U1(0)
      @test label(axes(b, 1)[Block(2)]) == U1(1)
      @test Array(a) isa Array{elt}
      @test Array(a) == a
      @test 2 * Array(a) == b
    end

    b = a[2:3, 2:3, 2:3, 2:3]
    @test size(b) == (2, 2, 2, 2)
    @test blocksize(b) == (2, 2, 2, 2)
    @test nstored(b) == 2
    @test block_nstored(b) == 2
    for i in 1:ndims(a)
      @test axes(b, i) isa GradedUnitRange
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
    # TODO: Once block merging is implemented, this should
    # be the real test.
    for ax in axes(m)
      @test ax isa GradedUnitRange
      # TODO: Current `fusedims` doesn't merge
      # common sectors, need to fix.
      @test_broken blocklabels(ax) == [U1(0), U1(1), U1(2)]
      @test blocklabels(ax) == [U1(0), U1(1), U1(1), U1(2)]
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
    # TODO: Current `fusedims` doesn't merge
    # common sectors, need to fix.
    @test_broken blocksize(m) == (3, 3)
    @test blocksize(m) == (4, 4)
    @test a == splitdims(m, (d1, d2), (d1, d2))
  end
  @testset "dual axes" begin
    r = gradedrange([U1(0) => 2, U1(1) => 2])
    a = BlockSparseArray{elt}(dual(r), r)
    @views for b in [Block(1, 1), Block(2, 2)]
      a[b] = randn(elt, size(a[b]))
    end
    # TODO: Define and use `isdual` here.
    @test axes(a, 1) isa UnitRangeDual
    @test axes(a, 2) isa GradedUnitRange
    @test !(axes(a, 2) isa UnitRangeDual)
    a_dense = Array(a)
    @test eachindex(a) == CartesianIndices(size(a))
    for I in eachindex(a)
      @test a[I] == a_dense[I]
    end
    @test axes(a') == dual.(reverse(axes(a)))
    # TODO: Define and use `isdual` here.
    @test axes(a', 1) isa UnitRangeDual
    @test axes(a', 2) isa GradedUnitRange
    @test !(axes(a', 2) isa UnitRangeDual)
    @test isnothing(show(devnull, MIME("text/plain"), a))

    # Check preserving dual in tensor algebra.
    for b in (a + a, 2 * a, 3 * a - a)
      @test Array(b) ≈ 2 * Array(a)
      # TODO: Define and use `isdual` here.
      @test axes(b, 1) isa UnitRangeDual
      @test axes(b, 2) isa GradedUnitRange
      @test !(axes(b, 2) isa UnitRangeDual)
    end

    @test isnothing(show(devnull, MIME("text/plain"), @view(a[Block(1, 1)])))
    @test @view(a[Block(1, 1)]) == a[Block(1, 1)]

    # Test case when all axes are dual.
    for r in (gradedrange([U1(0) => 2, U1(1) => 2]), blockedrange([2, 2]))
      a = BlockSparseArray{elt}(dual(r), dual(r))
      @views for i in [Block(1, 1), Block(2, 2)]
        a[i] = randn(elt, size(a[i]))
      end
      b = 2 * a
      @test block_nstored(b) == 2
      @test Array(b) == 2 * Array(a)
      for ax in axes(b)
        @test ax isa UnitRangeDual
      end
    end

    # Test case when all axes are dual
    # from taking the adjoint.
    for r in (gradedrange([U1(0) => 2, U1(1) => 2]), blockedrange([2, 2]))
      a = BlockSparseArray{elt}(r, r)
      @views for i in [Block(1, 1), Block(2, 2)]
        a[i] = randn(elt, size(a[i]))
      end
      b = 2 * a'
      @test block_nstored(b) == 2
      @test Array(b) == 2 * Array(a)'
      for ax in axes(b)
        @test ax isa UnitRangeDual
      end
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
