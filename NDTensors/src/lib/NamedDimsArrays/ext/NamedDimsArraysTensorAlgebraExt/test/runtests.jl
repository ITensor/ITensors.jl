@eval module $(gensym())
using Test: @test, @testset, @test_broken
using NDTensors.NamedDimsArrays: named, unname
using NDTensors.TensorAlgebra: TensorAlgebra
using LinearAlgebra: qr
@testset "NamedDimsArraysTensorAlgebraExt contract (eltype=$(elt))" for elt in (
  Float32, ComplexF32, Float64, ComplexF64
)
  i = named(2, "i")
  j = named(2, "j")
  k = named(2, "k")
  na1 = randn(elt, i, j)
  na2 = randn(elt, j, k)
  na_dest = TensorAlgebra.contract(na1, na2)
  @test eltype(na_dest) === elt
  @test unname(na_dest, (i, k)) ≈ unname(na1) * unname(na2)
end
@testset "NamedDimsArraysTensorAlgebraExt QR (eltype=$(elt))" for elt in (
  Float32, ComplexF32, Float64, ComplexF64
)
  di = 2
  dj = 2
  i = named(di, "i")
  j = named(dj, "j")
  na = randn(elt, i, j)
  @test_broken error("QR not implemented yet")
  # q, r = qr(na)
end
end
