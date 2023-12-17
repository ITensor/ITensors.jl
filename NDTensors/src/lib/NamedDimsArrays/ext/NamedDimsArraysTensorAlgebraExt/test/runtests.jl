@eval module $(gensym())
using Test: @test, @testset, @test_broken
using NDTensors.NamedDimsArrays: named, unname
using NDTensors.TensorAlgebra: TensorAlgebra, contract
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
  dims = (2, 2, 2, 2)
  i, j, k, l = named.(dims, ("i", "j", "k", "l"))

  na = randn(elt, i, j)
  # TODO: Should this be allowed?
  q, r = qr(na)
  @test q * r ≈ na

  na = randn(elt, i, j, k, l)
  q, r = qr(na, (i, k), (j, l))
  @test contract(q, r) ≈ na
end
end
