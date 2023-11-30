@eval module $(gensym())
using Combinatorics: permutations
using LinearAlgebra: qr
using NDTensors.TensorAlgebra: TensorAlgebra
using TensorOperations: TensorOperations
using Test: @test, @test_broken, @testset

@testset "TensorAlgebra" begin
  elts = (Float32, ComplexF32, Float64, ComplexF64)
  @testset "contract (eltype1=$elt1, eltype2=$elt2)" for elt1 in elts, elt2 in elts
    dims = (2, 3, 4, 5)
    labels = (:a, :b, :c, :d)
    for (d1s, d2s) in (((1, 2), (2, 3)), ((1, 2, 3), (2, 3, 4)), ((1, 2, 3), (3, 4)))
      a1 = randn(elt1, map(i -> dims[i], d1s))
      labels1 = map(i -> labels[i], d1s)
      a2 = randn(elt2, map(i -> dims[i], d2s))
      labels2 = map(i -> labels[i], d2s)
      for perm1 in permutations(1:ndims(a1)), perm2 in permutations(1:ndims(a2))
        a1′ = permutedims(a1, perm1)
        a2′ = permutedims(a2, perm2)
        labels1′ = map(i -> labels1[i], perm1)
        labels2′ = map(i -> labels2[i], perm2)
        a_dest, labels_dest = TensorAlgebra.contract(a1′, labels1′, a2′, labels2′)
        @test labels_dest == symdiff(labels1′, labels2′)
        a_dest_tensoroperations = TensorOperations.tensorcontract(
          labels_dest, a1′, labels1′, a2′, labels2′
        )
        @test a_dest ≈ a_dest_tensoroperations
      end
    end
  end
  @testset "contract broken" begin
    a1 = randn(3, 5, 8)
    a2 = randn(8, 2, 4)
    labels_dest = (:a, :b, :c, :d)
    labels1 = (:c, :a, :x)
    labels2 = (:x, :d, :b)
    @test_broken a′ = TensorAlgebra.contract(labels_dest, a1, labels1, a2, labels2)
  end
  @testset "qr" begin
    a = randn(5, 4, 3, 2)
    labels_a = (:a, :b, :c, :d)
    labels_q = (:b, :a)
    labels_r = (:d, :c)
    q, r = qr(a, labels_a, labels_q, labels_r)
    label_qr = :qr
    a′ = TensorAlgebra.contract(
      labels_a, q, (labels_q..., label_qr), r, (label_qr, labels_r...)
    )
    @test a ≈ a′
  end
end
end
