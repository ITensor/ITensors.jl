@eval module $(gensym())
using NDTensors: contract, tensor
using NDTensors.SparseArraysBase: densearray
using NDTensors.DiagonalArrays: DiagonalArray
using Test: @test, @testset
@testset "Tensor wrapping DiagonalArray" begin
  D = DiagonalArray(randn(3), 3, 4, 5)
  Dᵈ = densearray(D)
  A = randn(3, 4, 5)

  for convert_to_dense in (true, false)
    @test contract(D, (-1, -2, -3), A, (-1, -2, -3); convert_to_dense) ≈
      contract(Dᵈ, (-1, -2, -3), A, (-1, -2, -3))
    @test contract(D, (-1, -2, 1), A, (-1, -2, 2); convert_to_dense) ≈
      contract(Dᵈ, (-1, -2, 1), A, (-1, -2, 2))
  end

  # Tensor tests
  Dᵗ = tensor(D, size(D))
  Dᵈᵗ = tensor(Dᵈ, size(D))
  Aᵗ = tensor(A, size(A))
  @test contract(Dᵗ, (-1, -2, -3), Aᵗ, (-1, -2, -3)) ≈
    contract(Dᵈᵗ, (-1, -2, -3), Aᵗ, (-1, -2, -3))
end
end
