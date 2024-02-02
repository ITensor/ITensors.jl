module BlockSparseArraysTensorAlgebraExt
using BlockArrays: BlockedUnitRange
using ..BlockSparseArrays: AbstractBlockSparseArray, block_reshape
using ...GradedAxes: tensor_product
using ...TensorAlgebra: TensorAlgebra, FusionStyle, BlockReshapeFusion

TensorAlgebra.:⊗(a1::BlockedUnitRange, a2::BlockedUnitRange) = tensor_product(a1, a2)

TensorAlgebra.FusionStyle(::BlockedUnitRange) = BlockReshapeFusion()

function TensorAlgebra.fusedims(
  ::BlockReshapeFusion, a::AbstractArray, axes::AbstractUnitRange...
)
  return block_reshape(a, axes)
end

function TensorAlgebra.splitdims(
  ::BlockReshapeFusion, a::AbstractArray, axes::AbstractUnitRange...
)
  return block_reshape(a, axes)
end
end
