using Combinatorics: permutations
using NDTensors.TensorAlgebra: TensorAlgebra
using TensorOperations: TensorOperations
using Test: @test, @testset

@testset "TensorAlgebra" begin
  dims = (2, 3, 4, 5)
  labels = (:a, :b, :c, :d)
  for (d1s, d2s) in (((1, 2), (2, 3)), ((1, 2, 3), (2, 3, 4)), ((1, 2, 3), (3, 4)))
    a1 = randn(map(i -> dims[i], d1s))
    labels1 = map(i -> labels[i], d1s)
    a2 = randn(map(i -> dims[i], d2s))
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
