using ITensors
using LinearAlgebra
using Test

module TestArrayStorage
using ITensors
using ITensors.NDTensors.BlockSparseArrays
default_arraystoragetype(space) = Array
default_arraystoragetype(space::Vector{<:Pair{<:QN}}) = BlockSparseArray
is_qn_space(i) = false
is_qn_space(i::Vector{<:Pair{<:QN}}) = true
end

@testset "ITensor Array storage $space" for space in (2, [QN(0) => 2, QN(1) => 3])
  i, j, k, l = Index.((space, space, space, space))

  # TensorStorage
  A_ts = randomITensor(i, dag(j))
  B_ts = randomITensor(j, dag(k))
  Cij_ts = combiner(i, dag(j))
  Cik_ts = combiner(i, k)
  Ci_ts = combiner(i)
  C0_ts = combiner()

  A = NDTensors.to_arraystorage(A_ts)
  B = NDTensors.to_arraystorage(B_ts)
  Cij = NDTensors.to_arraystorage(Cij_ts)
  Cik = NDTensors.to_arraystorage(Cik_ts)
  Ci = NDTensors.to_arraystorage(Ci_ts)
  C0 = NDTensors.to_arraystorage(C0_ts)

  @test NDTensors.storage(A) isa TestArrayStorage.default_arraystoragetype(space)
  @test A + A ≈ A_ts + A_ts
  @test NDTensors.storage(A + A) isa TestArrayStorage.default_arraystoragetype(space)
  @test 2 * A ≈ 2 * A_ts
  @test NDTensors.storage(2A) isa TestArrayStorage.default_arraystoragetype(space)

  for (C, C_ts) in zip((Cij, Ci, C0), (Cij_ts, Ci_ts, C0))
    @test A * C ≈ A_ts * C_ts
    @test C * A ≈ A_ts * C_ts
    @test (A * C) * dag(C) ≈ A
    @test dag(C) * (A * C) ≈ A
    @test (C * A) * dag(C) ≈ A
    @test dag(C) * (C * A) ≈ A
    @test A * B ≈ A_ts * B_ts
  end

  # Partial combiner
  D_ts = randomITensor(i, j, k, l)
  D = NDTensors.to_arraystorage(D_ts)
  cik = uniqueind(Cik, D)

  @test D * Cik ≈ D_ts * Cik_ts
  @test permute(D * Cik, j, cik, l) * dag(Cik) ≈ D
  @test dag(Cik) * permute(D * Cik, j, cik, l) ≈ D
  @test permute(D * Cik, cik, j, l) * dag(Cik) ≈ D
  @test dag(Cik) * permute(D * Cik, cik, j, l) ≈ D
  @test permute(Cik * D, j, cik, l) * dag(Cik) ≈ D
  @test dag(Cik) * permute(Cik * D, j, cik, l) ≈ D
  @test permute(Cik * D, cik, j, l) * dag(Cik) ≈ D
  @test dag(Cik) * permute(Cik * D, cik, j, l) ≈ D

  # TODO: Still need to implement.
  if TestArrayStorage.is_qn_space(space)
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
