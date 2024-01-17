@eval module $(gensym())
using BlockArrays: Block
using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.GradedAxes: gradedrange
using NDTensors.Sectors: U1
using NDTensors.TensorAlgebra: contract
using Test: @test, @testset

function randn_blockdiagonal(elt::Type, axes...)
  a = BlockSparseArray{elt}(axes...)
end

const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "`contract` `BlockSparseArray` (eltype=$elt)" for elt in elts
  d = gradedrange([U1(0) => 2, U1(1) => 3])

  a1 = BlockSparseArray{elt}(d, d, d)
  a1[Block(1, 1, 1)] = randn(size(a1[Block(1, 1, 1)]))
  a1[Block(2, 2, 2)] = randn(size(a1[Block(2, 2, 2)]))

  a_dest, dimnames_dest = contract(a, (-1, 1, -2), a, (-1, -2, 2))
end
end
