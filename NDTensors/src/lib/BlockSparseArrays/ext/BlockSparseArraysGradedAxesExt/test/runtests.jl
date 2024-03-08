@eval module $(gensym())
using Test: @test, @testset, @test_broken
using BlockArrays: Block, blocksize
using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.GradedAxes: gradedrange
using NDTensors.Sectors: U1
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
    d1 = gradedrange([U1(0) => 1, U1(1) => 1])
    d2 = gradedrange([U1(1) => 1, U1(0) => 1])
    a = BlockSparseArray{elt}(d1, d2, d1, d2)
    blockdiagonal!(randn!, a)
    @test Array(a) isa Array{elt}
    @test Array(a) == a
    @test 2 * Array(a) == 2a
  end
  @testset "fusedims" begin
    d1 = gradedrange([U1(0) => 1, U1(1) => 1])
    d2 = gradedrange([U1(1) => 1, U1(0) => 1])
    a = BlockSparseArray{elt}(d1, d2, d1, d2)
    blockdiagonal!(randn!, a)
    m = fusedims(a, (1, 2), (3, 4))
    @test a[1, 1, 1, 1] == m[2, 2]
    @test a[2, 2, 2, 2] == m[3, 3]
    # TODO: Current `fusedims` doesn't merge
    # common sectors, need to fix.
    @test_broken blocksize(m) == (3, 3)
    @test a == splitdims(m, (d1, d2), (d1, d2))
  end
end
end
