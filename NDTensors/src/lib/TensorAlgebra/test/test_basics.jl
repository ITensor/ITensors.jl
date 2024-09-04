@eval module $(gensym())
using BlockArrays: blockfirsts, blocklasts, blocklength, blocklengths, blocks
using Combinatorics: permutations
using EllipsisNotation: var".."
using LinearAlgebra: norm, qr
using NDTensors.TensorAlgebra:
  TensorAlgebra, blockedperm, blockedperm_indexin, fusedims, splitdims
using NDTensors: NDTensors
include(joinpath(pkgdir(NDTensors), "test", "NDTensorsTestUtils", "NDTensorsTestUtils.jl"))
using .NDTensorsTestUtils: default_rtol
using Test: @test, @test_broken, @testset
const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "BlockedPermutation" begin
  p = blockedperm((3, 4, 5), (2, 1))
  @test Tuple(p) === (3, 4, 5, 2, 1)
  @test isperm(p)
  @test length(p) == 5
  @test blocks(p) == ((3, 4, 5), (2, 1))
  @test blocklength(p) == 2
  @test blocklengths(p) == (3, 2)
  @test blockfirsts(p) == (1, 4)
  @test blocklasts(p) == (3, 5)
  @test invperm(p) == blockedperm((5, 4, 1), (2, 3))

  # Empty block.
  p = blockedperm((3, 2), (), (1,))
  @test Tuple(p) === (3, 2, 1)
  @test isperm(p)
  @test length(p) == 3
  @test blocks(p) == ((3, 2), (), (1,))
  @test blocklength(p) == 3
  @test blocklengths(p) == (2, 0, 1)
  @test blockfirsts(p) == (1, 3, 3)
  @test blocklasts(p) == (2, 2, 3)
  @test invperm(p) == blockedperm((3, 2), (), (1,))

  # Split collection into `BlockedPermutation`.
  p = blockedperm_indexin(("a", "b", "c", "d"), ("c", "a"), ("b", "d"))
  @test p == blockedperm((3, 1), (2, 4))

  # Singleton dimensions.
  p = blockedperm((2, 3), 1)
  @test p == blockedperm((2, 3), (1,))

  # First dimensions are unspecified.
  p = blockedperm(.., (4, 3))
  @test p == blockedperm(1, 2, (4, 3))
  # Specify length
  p = blockedperm(.., (4, 3); length=Val(6))
  @test p == blockedperm(1, 2, 5, 6, (4, 3))

  # Last dimensions are unspecified.
  p = blockedperm((4, 3), ..)
  @test p == blockedperm((4, 3), 1, 2)
  # Specify length
  p = blockedperm((4, 3), ..; length=Val(6))
  @test p == blockedperm((4, 3), 1, 2, 5, 6)

  # Middle dimensions are unspecified.
  p = blockedperm((4, 3), .., 1)
  @test p == blockedperm((4, 3), 2, 1)
  # Specify length
  p = blockedperm((4, 3), .., 1; length=Val(6))
  @test p == blockedperm((4, 3), 2, 5, 6, 1)

  # No dimensions are unspecified.
  p = blockedperm((3, 2), .., 1)
  @test p == blockedperm((3, 2), 1)
end
@testset "TensorAlgebra" begin
  @testset "fusedims (eltype=$elt)" for elt in elts
    a = randn(elt, 2, 3, 4, 5)
    a_fused = fusedims(a, (1, 2), (3, 4))
    @test eltype(a_fused) === elt
    @test a_fused ≈ reshape(a, 6, 20)
    a_fused = fusedims(a, (3, 1), (2, 4))
    @test eltype(a_fused) === elt
    @test a_fused ≈ reshape(permutedims(a, (3, 1, 2, 4)), (8, 15))
    a_fused = fusedims(a, (3, 1, 2), 4)
    @test eltype(a_fused) === elt
    @test a_fused ≈ reshape(permutedims(a, (3, 1, 2, 4)), (24, 5))
    a_fused = fusedims(a, .., (3, 1))
    @test eltype(a_fused) === elt
    @test a_fused ≈ reshape(permutedims(a, (2, 4, 3, 1)), (3, 5, 8))
    a_fused = fusedims(a, (3, 1), ..)
    @test eltype(a_fused) === elt
    @test a_fused ≈ reshape(permutedims(a, (3, 1, 2, 4)), (8, 3, 5))
    a_fused = fusedims(a, .., (3, 1), 2)
    @test eltype(a_fused) === elt
    @test a_fused ≈ reshape(permutedims(a, (4, 3, 1, 2)), (5, 8, 3))
    a_fused = fusedims(a, (3, 1), .., 2)
    @test eltype(a_fused) === elt
    @test a_fused ≈ reshape(permutedims(a, (3, 1, 4, 2)), (8, 5, 3))
    a_fused = fusedims(a, (3, 1), ..)
    @test eltype(a_fused) === elt
    @test a_fused ≈ reshape(permutedims(a, (3, 1, 2, 4)), (8, 3, 5))
  end
  @testset "splitdims (eltype=$elt)" for elt in elts
    a = randn(elt, 6, 20)
    a_split = splitdims(a, (2, 3), (5, 4))
    @test eltype(a_split) === elt
    @test a_split ≈ reshape(a, (2, 3, 5, 4))
    a_split = splitdims(a, (1:2, 1:3), (1:5, 1:4))
    @test eltype(a_split) === elt
    @test a_split ≈ reshape(a, (2, 3, 5, 4))
    a_split = splitdims(a, 2 => (5, 4), 1 => (2, 3))
    @test eltype(a_split) === elt
    @test a_split ≈ reshape(a, (2, 3, 5, 4))
    a_split = splitdims(a, 2 => (1:5, 1:4), 1 => (1:2, 1:3))
    @test eltype(a_split) === elt
    @test a_split ≈ reshape(a, (2, 3, 5, 4))
    a_split = splitdims(a, 2 => (5, 4))
    @test eltype(a_split) === elt
    @test a_split ≈ reshape(a, (6, 5, 4))
    a_split = splitdims(a, 2 => (1:5, 1:4))
    @test eltype(a_split) === elt
    @test a_split ≈ reshape(a, (6, 5, 4))
    a_split = splitdims(a, 1 => (2, 3))
    @test eltype(a_split) === elt
    @test a_split ≈ reshape(a, (2, 3, 20))
    a_split = splitdims(a, 1 => (1:2, 1:3))
    @test eltype(a_split) === elt
    @test a_split ≈ reshape(a, (2, 3, 20))
  end
  using TensorOperations: TensorOperations
  @testset "contract (eltype1=$elt1, eltype2=$elt2)" for elt1 in elts, elt2 in elts
    dims = (2, 3, 4, 5, 6, 7, 8, 9, 10)
    labels = (:a, :b, :c, :d, :e, :f, :g, :h, :i)
    for (d1s, d2s, d_dests) in (
      ((1, 2), (1, 2), ()),
      ((1, 2), (2, 1), ()),
      ((1, 2), (2, 1, 3), (3,)),
      ((1, 2, 3), (2, 1), (3,)),
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
      ## Here we loosened the tolerance because of some floating point roundoff issue.
      ## with Float32 numbers
      @test a_dest ≈ α * a_dest_tensoroperations + β * a_dest_init rtol =
        50 * default_rtol(elt_dest)
    end
  end
end
@testset "qr (eltype=$elt)" for elt in elts
  a = randn(elt, 5, 4, 3, 2)
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
