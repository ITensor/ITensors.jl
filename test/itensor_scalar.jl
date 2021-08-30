using ITensors
using Test

@testset "Scalar ITensors" begin
  A = ITensor(2.4)
  @test storage(A) isa ITensors.Dense{Float64}
  @test ndims(A) == 0
  @test order(A) == 0
  @test A[] == 2.4
  @test A[1] == 2.4
  @test scalar(A) == 2.4
  @test ITensors.symmetrystyle(A) == ITensors.NonQN()

  A[] = 3.4
  @test ndims(A) == 0
  @test order(A) == 0
  @test A[] == 3.4
  @test A[1] == 3.4
  @test scalar(A) == 3.4
  @test ITensors.symmetrystyle(A) == ITensors.NonQN()

  A[1] = 4.4
  @test ndims(A) == 0
  @test order(A) == 0
  @test A[] == 4.4
  @test A[1] == 4.4
  @test scalar(A) == 4.4
  @test ITensors.symmetrystyle(A) == ITensors.NonQN()

  A = ITensor()
  @test storage(A) isa ITensors.EmptyStorage{ITensors.EmptyNumber}
  @test ndims(A) == 0
  @test order(A) == 0
  @test A[] == 0.0
  @test A[1] == 0.0
  @test scalar(A) == 0.0
  @test ITensors.symmetrystyle(A) == ITensors.NonQN()

  A = ITensor()
  @test storage(A) isa ITensors.EmptyStorage{ITensors.EmptyNumber}
  A[] = 3.4
  @test storage(A) isa ITensors.Dense{Float64}
  @test ndims(A) == 0
  @test order(A) == 0
  @test A[] == 3.4
  @test A[1] == 3.4
  @test scalar(A) == 3.4
  @test ITensors.symmetrystyle(A) == ITensors.NonQN()

  A = ITensor()
  @test storage(A) isa ITensors.EmptyStorage{ITensors.EmptyNumber}
  A[1] = 4.4
  @test storage(A) isa ITensors.Dense{Float64}
  @test ndims(A) == 0
  @test order(A) == 0
  @test A[] == 4.4
  @test A[1] == 4.4
  @test scalar(A) == 4.4
  @test ITensors.symmetrystyle(A) == ITensors.NonQN()

  x = 2.3
  ITensor(fill(x, ())) == ITensor(x)
  ITensor(fill(x, (1))) == ITensor(x)
  ITensor(fill(x, (1, 1))) == ITensor(x)
  ITensor(fill(x, (1, 1, 1))) == ITensor(x)
  @test_throws ErrorException ITensor(fill(x, (2, 2)))
end

nothing
