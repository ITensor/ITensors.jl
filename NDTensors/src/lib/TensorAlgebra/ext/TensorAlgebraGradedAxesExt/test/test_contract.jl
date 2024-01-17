@eval module $(gensym())
using BlockArrays: Block
using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.GradedAxes: gradedrange
using NDTensors.Sectors: U1
using NDTensors.TensorAlgebra: contract

d = gradedrange([U1(0) => 2, U1(1) => 3])
# d = [2, 3]

# TODO: Fix `BlockSparseArray{Float64}(d, d, d)`.
a = BlockSparseArray{Float64}((d, d, d))
a[Block(1, 1, 1)] = randn(size(a[Block(1, 1, 1)]))
a[Block(2, 2, 2)] = randn(size(a[Block(2, 2, 2)]))

a_dest, dimnames_dest = contract(a, (1, -1, -2), a, (-1, -2, 2))
