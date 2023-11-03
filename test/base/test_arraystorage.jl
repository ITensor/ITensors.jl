using ITensors
using LinearAlgebra
using Test

module TestArrayStorage
using ITensors
using ITensors.NDTensors.BlockSparseArrays
default_arraystoragetype(space) = Array
default_arraystoragetype(space::Vector{<:Pair{<:QN}}) = BlockSparseArray
end

is_qn_space(i) = false
is_qn_space(i::Vector{<:Pair{<:QN}}) = true

@testset "ITensor Array storage $space" for space in (2, [QN(0) => 2, QN(1) => 3])
  i, j, k = Index.((space, space, space))

  # TensorStorage
  A_ts = randomITensor(i, dag(j))
  B_ts = randomITensor(j, dag(k))
  C_ts = combiner(i, dag(j))

  A = NDTensors.to_arraystorage(A_ts)
  B = NDTensors.to_arraystorage(B_ts)
  C = NDTensors.to_arraystorage(C_ts)

  @test NDTensors.storage(A) isa TestArrayStorage.default_arraystoragetype(space)
  @test A + A ≈ A_ts + A_ts
  @test NDTensors.storage(A + A) isa TestArrayStorage.default_arraystoragetype(space)
  @test 2 * A ≈ 2 * A_ts
  @test NDTensors.storage(2A) isa TestArrayStorage.default_arraystoragetype(space)

  # TODO: Test combining over subset of indices.
  @test A * C ≈ A_ts * C_ts
  if is_qn_space(space)
    @test_broken (A * C) * dag(C) ≈ A
  else
    @test (A * C) * dag(C) ≈ A
  end
  @test A * B ≈ A_ts * B_ts

  # Partial combiner
  D_ts = randomITensor(i, j, k)
  Cᴰ_ts = combiner(i, k)
  D = NDTensors.to_arraystorage(D_ts)
  Cᴰ = NDTensors.to_arraystorage(Cᴰ_ts)
  @test D * Cᴰ ≈ D_ts * Cᴰ_ts

  # TODO: Still need to implement.
  if is_qn_space(space)
    @test_broken NDTensors.storage(A * B) isa
      TestArrayStorage.default_arraystoragetype(space)
    @test_broken A[1, 1] = 11
  else
    @test NDTensors.storage(A * B) isa TestArrayStorage.default_arraystoragetype(space)
    A[1, 1] = 11
    @test A[1, 1] == 11
  end
  @test_broken qr(A, i)
  @test_broken eigen(A, i)
  @test_broken svd(A, i)
end
