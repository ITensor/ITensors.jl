@eval module $(gensym())
using Test: @test, @testset
using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.TensorAlgebra: contract
using NDTensors.SparseArraysBase: densearray
@testset "BlockSparseArraysTensorAlgebraExt (eltype=$elt)" for elt in (
  Float32, Float64, Complex{Float32}, Complex{Float64}
)
  a1 = BlockSparseArray{elt}([1, 2], [2, 3], [3, 2])
  a2 = BlockSparseArray{elt}([2, 2], [3, 2], [2, 3])
  a_dest, dimnames_dest = contract(a1, (1, -1, -2), a2, (2, -2, -1))
  a_dest_dense, dimnames_dest_dense = contract(
    densearray(a1), (1, -1, -2), densearray(a2), (2, -2, -1)
  )
  @test a_dest ≈ a_dest_dense
end
end
