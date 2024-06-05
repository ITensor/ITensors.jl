@eval module $(gensym())
using Compat: Returns
using Test: @test, @testset, @test_broken
using BlockArrays: Block, blocklength, blocksize
using NDTensors.BlockSparseArrays: BlockSparseArray, block_nstored
using NDTensors.GradedAxes: GradedAxes, GradedUnitRange, UnitRangeDual, dual, gradedrange
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
    @test blocksize(b) == (1, 1, 1, 1)
    @test nstored(b) == length(b)
    @test block_nstored(b) == blocklength(b)
    for i in 1:ndims(a)
      @test axes(b, i) isa Base.OneTo{Int}
    end
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
    @test axes(m, 1) isa GradedUnitRange
    @test axes(m, 2) isa GradedUnitRange
    @test a[1, 1, 1, 1] == m[1, 1]
    @test a[2, 2, 2, 2] == m[4, 4]
    # TODO: Current `fusedims` doesn't merge
    # common sectors, need to fix.
    @test_broken blocksize(m) == (3, 3)
    @test a == splitdims(m, (d1, d2), (d1, d2))
  end
  @testset "dual axes" begin
    r = gradedrange([U1(0) => 2, U1(1) => 2])
    a = BlockSparseArray{elt}(dual(r), r)
    a[Block(1, 1)] = randn(elt, size(a[Block(1, 1)]))
    a[Block(2, 2)] = randn(elt, size(a[Block(2, 2)]))
    a_dense = Array(a)
    @test eachindex(a) == CartesianIndices(size(a))
    for I in eachindex(a)
      @test a[I] == a_dense[I]
    end
    @test axes(a') == dual.(reverse(axes(a)))
    # TODO: Define and use `isdual` here.
    @test axes(a', 1) isa UnitRangeDual
    @test !(axes(a', 2) isa UnitRangeDual)
    @test isnothing(show(devnull, MIME("text/plain"), a))
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
