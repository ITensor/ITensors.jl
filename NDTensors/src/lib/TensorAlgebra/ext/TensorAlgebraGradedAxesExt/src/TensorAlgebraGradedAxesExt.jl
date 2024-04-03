module TensorAlgebraGradedAxesExt
using ...GradedAxes: GradedUnitRange, tensor_product
using ..TensorAlgebra: TensorAlgebra

function TensorAlgebra.:⊗(a1::GradedUnitRange, a2::GradedUnitRange)
  return tensor_product(a1, a2)
end
end
