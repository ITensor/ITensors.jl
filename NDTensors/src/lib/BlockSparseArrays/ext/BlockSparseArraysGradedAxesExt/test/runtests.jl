@eval module $(gensym())
using Compat: Returns
using Test: @test, @testset, @test_broken
using BlockArrays: Block, blocksize
using NDTensors.BlockSparseArrays: BlockSparseArray, block_nstored
using NDTensors.GradedAxes: GradedUnitRange, gradedrange
using NDTensors.LabelledNumbers: label
using NDTensors.Sectors: U1
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
      if VERSION >= v"1.7"
        # TODO: Have to investigate why this fails
        # on Julia v1.6, or drop support for v1.6.
        for i in 1:ndims(a)
          @test axes(b, i) isa GradedUnitRange
        end
        @test label(axes(b, 1)[Block(1)]) == U1(0)
        @test label(axes(b, 1)[Block(2)]) == U1(1)
      end
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
    @test axes(m, 1) isa GradedUnitRange
    @test axes(m, 2) isa GradedUnitRange
    @test a[1, 1, 1, 1] == m[1, 1]
    @test a[2, 2, 2, 2] == m[4, 4]
    # TODO: Current `fusedims` doesn't merge
    # common sectors, need to fix.
    @test_broken blocksize(m) == (3, 3)
    @test a == splitdims(m, (d1, d2), (d1, d2))
  end
end
end
