using NDTensors.TensorAlgebra: TensorAlgebra
using Test: @test, @testset

@testset "TensorAlgebra" begin
  d1, d2, d3 = (2, 3, 4)
  a = randn(d1, d2)
  b = randn(d2, d3)
  labels_a = (1, -1)
  labels_b = (-1, 2)
  c, labels_c = TensorAlgebra.contract(a, labels_a, b, labels_b)
end
