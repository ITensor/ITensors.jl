using Test: @test, @testset
using NDTensors.NamedDimsArrays: named, unname
using NDTensors.TensorAlgebra: TensorAlgebra

@testset "NamedDimsArraysTensorAlgebraExt" begin
  i = named(2, "i")
  j = named(2, "j")
  k = named(2, "k")
  na1 = randn(i, j)
  na2 = randn(j, k)
  na_dest = TensorAlgebra.contract(na1, na2)
  @test unname(na_dest, (i, k)) ≈ unname(na1) * unname(na2)
end
