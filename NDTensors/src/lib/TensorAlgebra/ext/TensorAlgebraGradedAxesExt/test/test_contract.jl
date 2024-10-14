@eval module $(gensym())
using BlockArrays: Block, blocksize
using Compat: Returns
using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.GradedAxes: gradedrange
using NDTensors.SparseArrayInterface: densearray
using NDTensors.SymmetrySectors: U1
using NDTensors.TensorAlgebra: contract
using Random: randn!
using Test: @test, @testset

function randn_blockdiagonal(elt::Type, axes::Tuple)
  a = BlockSparseArray{elt}(axes)
  blockdiaglength = minimum(blocksize(a))
  for i in 1:blockdiaglength
    b = Block(ntuple(Returns(i), ndims(a)))
    a[b] = randn!(a[b])
  end
  return a
end

const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "`contract` `BlockSparseArray` (eltype=$elt)" for elt in elts
  d = gradedrange([U1(0) => 2, U1(1) => 3])
  a1 = randn_blockdiagonal(elt, (d, d, d))
  a2 = randn_blockdiagonal(elt, (d, d, d))
  a_dest, dimnames_dest = contract(a1, (-1, 1, -2), a2, (-1, -2, 2))
  a1_dense = densearray(a1)
  a2_dense = densearray(a2)
  a_dest_dense, dimnames_dest_dense = contract(a1_dense, (-1, 1, -2), a2_dense, (-1, -2, 2))
  @test dimnames_dest == dimnames_dest_dense
  @test a_dest ≈ a_dest_dense
end
end
