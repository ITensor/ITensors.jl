module BlockSparseArraysTensorAlgebraExt
using BlockArrays: BlockedUnitRange
using ..BlockSparseArrays: AbstractBlockSparseArray, block_reshape
using ...GradedAxes: tensor_product
using ...TensorAlgebra: TensorAlgebra

TensorAlgebra.:⊗(a1::BlockedUnitRange, a2::BlockedUnitRange) = tensor_product(a1, a2)

function TensorAlgebra.fusedims(a::AbstractBlockSparseArray, axes::AbstractUnitRange...)
  return block_reshape(a, axes)
end

function TensorAlgebra.splitdims(a::AbstractBlockSparseArray, axes::AbstractUnitRange...)
  return block_reshape(a, axes)
end
end
