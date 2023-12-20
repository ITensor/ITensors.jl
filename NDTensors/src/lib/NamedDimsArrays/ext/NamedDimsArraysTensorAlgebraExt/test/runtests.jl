@eval module $(gensym())
using Test: @test, @testset, @test_broken
using NDTensors.NamedDimsArrays: named, unname
using NDTensors.TensorAlgebra: TensorAlgebra, contract, fusedims
using LinearAlgebra: qr
elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "NamedDimsArraysTensorAlgebraExt (eltype=$(elt))" for elt in elts
  @testset "contract" begin
    i = named(2, "i")
    j = named(2, "j")
    k = named(2, "k")
    na1 = randn(elt, i, j)
    na2 = randn(elt, j, k)
    na_dest = TensorAlgebra.contract(na1, na2)
    @test eltype(na_dest) === elt
    @test unname(na_dest, (i, k)) ≈ unname(na1) * unname(na2)
  end
  @testset "fusedims" begin
    i, j, k, l = named.((2, 3, 4, 5), ("i", "j", "k", "l"))
    na = randn(elt, i, j, k, l)
    na_fused = fusedims(na, (k, i) => "a", (j, l) => "b")
    @test unname(na_fused, ("a", "b")) ≈
      reshape(unname(na, (k, i, j, l)), (unname(k) * unname(i), unname(j) * unname(l)))
  end
  @testset "qr" begin
    dims = (2, 2, 2, 2)
    i, j, k, l = named.(dims, ("i", "j", "k", "l"))

    na = randn(elt, i, j)
    # TODO: Should this be allowed?
    # TODO: Add support for specifying new name.
    q, r = qr(na)
    @test q * r ≈ na

    na = randn(elt, i, j, k, l)
    # TODO: Add support for specifying new name.
    q, r = qr(na, (i, k), (j, l))
    @test contract(q, r) ≈ na
  end
end
end
