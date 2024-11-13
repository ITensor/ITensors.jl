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
  blockaxes,
  mortar
using LinearAlgebra: Adjoint, mul!, norm
using NDTensors.BlockSparseArrays:
  @view!,
  BlockSparseArray,
  BlockView,
  block_nstored,
  block_reshape,
  block_stored_indices,
  view!
using NDTensors.SparseArrayInterface: nstored
using NDTensors.TensorAlgebra: contract

using Test

T = Float64

# scalar indexing
a = BlockSparseArray{T}([2, 3], [2, 2])
for i in blockaxes(a, 1), j in blockaxes(a, 2)
  a[i, j] = randn(T, blocksizes(a)[Int(i), Int(j)])
end
a

a[1, 2]
a[Block(1, 1)]
a[Block.(1:2), Block(1)]
aslice = a[Block.(1:2), 1]
axes(aslice, 1)
axes(aslice, 2)
length(axes(aslice)) == ndims(aslice)

aslice = a[Block.(1:2), 1:3]
axes(aslice)

mask = trues(size(a, 2))
aslice = a[:, mask]
aslice = a[:, [1, 2]]

a[Block(1, 1)] = randn(T, 2, 2)
a[Block(2, 2)] = randn(T, 2, 2)
