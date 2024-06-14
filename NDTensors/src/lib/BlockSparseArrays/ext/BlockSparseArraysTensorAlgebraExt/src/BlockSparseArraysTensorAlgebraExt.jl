module BlockSparseArraysTensorAlgebraExt
using BlockArrays: AbstractBlockedUnitRange
using ..BlockSparseArrays: AbstractBlockSparseArray, block_reshape
using ...GradedAxes: tensor_product
using ...TensorAlgebra: TensorAlgebra, FusionStyle, BlockReshapeFusion

function TensorAlgebra.:⊗(a1::AbstractBlockedUnitRange, a2::AbstractBlockedUnitRange)
  return tensor_product(a1, a2)
end

TensorAlgebra.FusionStyle(::AbstractBlockedUnitRange) = BlockReshapeFusion()

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
