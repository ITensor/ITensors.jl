using ITensors
using Test

@testset "Empty ITensor storage operations" begin
  i, j, k = Index.(2, ("i", "j", "k"))

  A = ITensor(i, j)
  B = ITensor(j, k)

  @test norm(A) == 0.0
  @test norm(B) == 0.0

  C = A * B
  @test hassameinds(C, (i, k))
  @test NDTensors.is_unallocated_zeros(C)

  A = ITensor(Float64, i, j)
  B = ITensor(j, k)
  C = A * B
  @test hassameinds(C, (i, k))
  @test NDTensors.is_unallocated_zeros(C)
  @test eltype(C) == Float64

  A = ITensor(i, j)
  B = ITensor(ComplexF64, j, k)

  @test norm(A) == 0.0
  @test norm(B) == 0.0
  @test norm(B) isa Float64

  C = A * B
  @test hassameinds(C, (i, k))
  @test NDTensors.is_unallocated_zeros(C)
  @test eltype(C) == ComplexF64

  A = ITensor(Float64, i, j)
  B = ITensor(ComplexF64, j, k)
  C = A * B
  @test hassameinds(C, (i, k))
  @test NDTensors.is_unallocated_zeros(C)
  @test eltype(C) == ComplexF64
end

@testset "Empty ITensor storage addition" begin
  i, j = Index.((2, 3))

  A = ITensor(i, j)
  B = randomITensor(j, i)

  ## if A is unallocated then B will be returned and
  ## the indices will be equivalent to `j, i` We 
  ## can permute but thats extra work.
  C = A + B
  @test inds(C) == (j, i)
  @test C ≈ B

  C = B + A
  @test inds(C) == (j, i)
  @test C ≈ B
end

@testset "Empty QN ITensor storage operations" begin
  i = Index([QN(0) => 1, QN(1) => 1])
  A = ITensor(i', dag(i))

  @test NDTensors.is_unallocated_zeros(A)

  C = A' * A

  @test hassameinds(C, (i'', i))
  storage(C)
  storage(C)
  @test storage(C) isa BlockSparse{NDTensors.UnspecifiedZero, NDTensors.UnallocatedZeros{NDTensors.UnspecifiedZero, 1, Tuple{Base.OneTo{Int64}}, Vector{NDTensors.UnspecifiedZero}}, 2}

  B = randomITensor(dag(i), i')

  C = A' * B

  @test hassameinds(C, (i'', i))
  @test storage(C) isa BlockSparse{Float64, NDTensors.UnallocatedZeros{Float64, 1, Tuple{Base.OneTo{Int64}}, Vector{Float64}}, 2}

  C = B' * A

  @test hassameinds(C, (i'', i))
  @test storage(C) isa  BlockSparse{Float64, NDTensors.UnallocatedZeros{Float64, 1, Tuple{Base.OneTo{Int64}}, Vector{Float64}}, 2}

  C = B + A
  @test inds(C) == inds(B)
  @test C ≈ B

  C = A + B
  @test C ≈ B
end

@testset "blockoffsets" for space in (2, [QN(0) => 1, QN(1) => 1])
  i = Index(space)
  A = ITensor(i', dag(i))
  @test blockoffsets(A) == NDTensors.BlockOffsets{2}()
end

@testset "zero" for space in (2, [QN(0) => 1, QN(1) => 1])
  i = Index(space)
  A = ITensor(i', dag(i))
  @test NDTensors.tensor(zero(A)) isa typeof(NDTensors.tensor(A))
end

nothing
