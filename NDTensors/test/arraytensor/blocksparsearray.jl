using NDTensors
using NDTensors.BlockSparseArrays
using BlockArrays: BlockArrays
using LinearAlgebra
using Test

using NDTensors: storage, storagetype

@testset "Tensor wrapping BlockSparseArray" begin
  is1 = ([1, 1], [1, 2])
  D1 = BlockSparseArray(
    [BlockArrays.Block(1, 1), BlockArrays.Block(2, 2)], [randn(1, 1), randn(1, 2)], is1
  )

  is2 = ([1, 2], [2, 2])
  D2 = BlockSparseArray(
    [BlockArrays.Block(1, 1), BlockArrays.Block(2, 2)], [randn(1, 2), randn(2, 2)], is2
  )

  T1 = tensor(D1, is1)
  T2 = tensor(D2, is2)

  @test T1[1, 1] == D1[1, 1]

  x = rand()
  T1[1, 1] = x

  @test T1[1, 1] == x
  @test array(T1) == D1
  @test storagetype(T1) <: BlockSparseArray{Float64,2}
  @test storage(T1) == D1
  @test eltype(T1) == eltype(D1)
  @test inds(T1) == is1

  @test_broken R = T1 * T2
  @test_broken storagetype(R) <: Matrix{Float64}
  @test_broken Array(R) ≈ Array(T1) * Array(T2)

  @test_broken T1r = randn!(similar(T1))
  @test_broken Array(T1r + T1) ≈ Array(T1r) + Array(T1)
  @test_broken Array(permutedims(T1, (2, 1))) ≈ permutedims(Array(T1), (2, 1))

  # TODO: Not implemented yet.
  ## U, S, V = svd(T1)
  ## @test U * S * V ≈ T1

  @test_broken T12 = contract(T1, (1, -1), T2, (-1, 2))
  @test_broken T12 ≈ T1 * T2

  @test_broken D12 = contract(D1, (1, -1), D2, (-1, 2))
  @test_broken D12 ≈ Array(T12)
end
