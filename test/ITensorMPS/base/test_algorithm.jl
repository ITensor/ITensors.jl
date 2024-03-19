using ITensors
using Test

@testset "Algorithm" begin
  alg = ITensors.Algorithm("X")

  @test alg isa ITensors.Algorithm"X"
  @test alg == ITensors.Algorithm"X"()

  s = siteinds("S=1/2", 4)
  A = MPO(s, "Id")
  ψ = randomMPS(s)

  @test_throws MethodError contract(alg, A, ψ)
  @test_throws MethodError contract(A, ψ; method="X")
  @test_throws MethodError contract(A, ψ; alg="X")
  @test contract(ITensors.Algorithm("densitymatrix"), A, ψ) ≈ A * ψ
  @test contract(ITensors.Algorithm("naive"), A, ψ) ≈ A * ψ
  @test contract(A, ψ; alg="densitymatrix") ≈ A * ψ
  @test contract(A, ψ; method="densitymatrix") ≈ A * ψ
  @test contract(A, ψ; alg="naive") ≈ A * ψ
  @test contract(A, ψ; method="naive") ≈ A * ψ

  B = copy(A)
  truncate!(ITensors.Algorithm("frobenius"), B)
  @test A ≈ B

  B = copy(A)
  truncate!(B; alg="frobenius")
  @test A ≈ B

  # Custom algorithm
  function ITensors.truncate!(::ITensors.Algorithm"my_new_algorithm", A::MPO; cutoff=1e-15)
    return "my_new_algorithm was called with cutoff $cutoff"
  end
  cutoff = 1e-5
  res = truncate!(A; alg="my_new_algorithm", cutoff=cutoff)
  @test res == "my_new_algorithm was called with cutoff $cutoff"
end
