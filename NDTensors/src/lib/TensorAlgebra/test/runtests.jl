@eval module $(gensym())
using Combinatorics: permutations
using LinearAlgebra: norm, qr
using NDTensors.TensorAlgebra: TensorAlgebra
using NDTensors: NDTensors
include(joinpath(pkgdir(NDTensors), "test", "NDTensorsTestUtils", "NDTensorsTestUtils.jl"))
using .NDTensorsTestUtils: default_rtol
using TensorOperations: TensorOperations
using Test: @test, @test_broken, @testset

@testset "TensorAlgebra" begin
  elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
  @testset "contract (eltype1=$elt1, eltype2=$elt2)" for elt1 in elts, elt2 in elts
    dims = (2, 3, 4, 5, 6, 7, 8, 9, 10)
    labels = (:a, :b, :c, :d, :e, :f, :g, :h, :i)
    for (d1s, d2s, d_dests) in (
      ((1, 2), (2, 3), (1, 3)),
      ((1, 2), (2, 3), (3, 1)),
      ((2, 1), (2, 3), (3, 1)),
      ((1, 2, 3), (2, 3, 4), (1, 4)),
      ((1, 2, 3), (2, 3, 4), (4, 1)),
      ((3, 2, 1), (4, 2, 3), (4, 1)),
      ((1, 2, 3), (3, 4), (1, 2, 4)),
      ((1, 2, 3), (3, 4), (4, 1, 2)),
      ((1, 2, 3), (3, 4), (2, 4, 1)),
      ((3, 1, 2), (3, 4), (2, 4, 1)),
      ((3, 2, 1), (4, 3), (2, 4, 1)),
      ((1, 2, 3, 4, 5, 6), (4, 5, 6, 7, 8, 9), (1, 2, 3, 7, 8, 9)),
      ((2, 4, 5, 1, 6, 3), (6, 4, 9, 8, 5, 7), (1, 7, 2, 8, 3, 9)),
    )
      a1 = randn(elt1, map(i -> dims[i], d1s))
      labels1 = map(i -> labels[i], d1s)
      a2 = randn(elt2, map(i -> dims[i], d2s))
      labels2 = map(i -> labels[i], d2s)
      labels_dest = map(i -> labels[i], d_dests)

      # Don't specify destination labels
      a_dest, labels_dest′ = TensorAlgebra.contract(a1, labels1, a2, labels2)
      a_dest_tensoroperations = TensorOperations.tensorcontract(
        labels_dest′, a1, labels1, a2, labels2
      )
      @test a_dest ≈ a_dest_tensoroperations

      # Specify destination labels
      a_dest = TensorAlgebra.contract(labels_dest, a1, labels1, a2, labels2)
      a_dest_tensoroperations = TensorOperations.tensorcontract(
        labels_dest, a1, labels1, a2, labels2
      )
      @test a_dest ≈ a_dest_tensoroperations

      # Specify α and β
      elt_dest = promote_type(elt1, elt2)
      # TODO: Using random `α`, `β` causing
      # random test failures, investigate why.
      α = elt_dest(1.2) # randn(elt_dest)
      β = elt_dest(2.4) # randn(elt_dest)
      a_dest_init = randn(elt_dest, map(i -> dims[i], d_dests))
      a_dest = copy(a_dest_init)
      TensorAlgebra.contract!(a_dest, labels_dest, a1, labels1, a2, labels2, α, β)
      a_dest_tensoroperations = TensorOperations.tensorcontract(
        labels_dest, a1, labels1, a2, labels2
      )
      @test a_dest ≈ α * a_dest_tensoroperations + β * a_dest_init rtol = default_rtol(
        elt_dest
      )
    end
  end
  @testset "qr" begin
    # a = randn(5, 4, 3, 2)
    a = randn(2, 2, 2, 2)
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
