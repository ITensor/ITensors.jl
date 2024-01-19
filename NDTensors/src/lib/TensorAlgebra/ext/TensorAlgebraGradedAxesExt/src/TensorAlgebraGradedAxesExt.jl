module TensorAlgebraGradedAxesExt
using ...GradedAxes: AbstractGradedUnitRange, tensor_product
using ..TensorAlgebra: TensorAlgebra

function TensorAlgebra.:⊗(a1::AbstractGradedUnitRange, a2::AbstractGradedUnitRange)
  return tensor_product(a1, a2)
end
end
