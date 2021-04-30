using ITensors
using Test

@testset "Empty ITensor storage operations" begin
  i, j, k = Index.(2, ("i", "j", "k"))

  A = ITensor(i, j)
  B = ITensor(j, k)
  C = A * B
  @test hassameinds(C, (i, k))
  @test storage(C) isa ITensors.EmptyStorage{ITensors.EmptyNumber,<:ITensors.Dense{ITensors.EmptyNumber}}

  A = ITensor(Float64, i, j)
  B = ITensor(j, k)
  C = A * B
  @test hassameinds(C, (i, k))
  @test storage(C) isa ITensors.EmptyStorage{Float64,<:ITensors.Dense{Float64}}

  A = ITensor(i, j)
  B = ITensor(ComplexF64, j, k)
  C = A * B
  @test hassameinds(C, (i, k))
  @test storage(C) isa ITensors.EmptyStorage{ComplexF64,<:ITensors.Dense{ComplexF64}}

  A = ITensor(Float64, i, j)
  B = ITensor(ComplexF64, j, k)
  C = A * B
  @test hassameinds(C, (i, k))
  @test storage(C) isa ITensors.EmptyStorage{ComplexF64,<:ITensors.Dense{ComplexF64}}
end

