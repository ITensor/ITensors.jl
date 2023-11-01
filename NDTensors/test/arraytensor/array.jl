using NDTensors
using LinearAlgebra
using Test

using NDTensors: storage, storagetype

@testset "Tensor wrapping Array" begin
  is1 = (2, 3)
  D1 = randn(is1)

  is2 = (3, 4)
  D2 = randn(is2)

  T1 = tensor(D1, is1)
  T2 = tensor(D2, is2)

  @test T1[1, 1] == D1[1, 1]

  x = rand()
  T1[1, 1] = x

  @test T1[1, 1] == x
  @test array(T1) == D1
  @test storagetype(T1) <: Matrix{Float64}
  @test storage(T1) == D1
  @test eltype(T1) == eltype(D1)
  @test inds(T1) == is1

  R = T1 * T2
  @test storagetype(R) <: Matrix{Float64}
  @test Array(R) ≈ Array(T1) * Array(T2)

  T1r = randn!(similar(T1))
  @test Array(T1r + T1) ≈ Array(T1r) + Array(T1)
  @test Array(permutedims(T1, (2, 1))) ≈ permutedims(Array(T1), (2, 1))

  U, S, V = svd(T1)

  # TODO: Should this work? Currently broken.
  ## I was able to fix this test but labels have to match up
  ## If you do U * S * V it fails because (U * S) is (2,2) and V is (3,2)
  @test U * S * V' ≈ T1
  # TODO: Should this require labels, or use existing labels?
  @test contract(contract(U, (1, -1), S, (-1, 2)), (1, -1), V, (2, -1)) ≈ T1

  T12 = contract(T1, (1, -1), T2, (-1, 2))
  @test T12 ≈ T1 * T2

  D12 = contract(D1, (1, -1), D2, (-1, 2))
  @test D12 ≈ Array(T12)
end
