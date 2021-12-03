using NDTensors
using LinearAlgebra
using Test

@testset "random_orthog" begin
  n, m = 10, 4
  O1 = random_orthog(n, m)
  @test eltype(O1) == Float64
  @test norm(transpose(O1) * O1 - Diagonal(fill(1.0, m))) < 1E-14
  O2 = random_orthog(m, n)
  @test norm(O2 * transpose(O2) - Diagonal(fill(1.0, m))) < 1E-14
end

@testset "random_unitary" begin
  n, m = 10, 4
  U1 = random_unitary(n, m)
  @test eltype(U1) == ComplexF64
  @test norm(U1' * U1 - Diagonal(fill(1.0, m))) < 1E-14
  U2 = random_unitary(m, n)
  @test norm(U2 * U2' - Diagonal(fill(1.0, m))) < 1E-14
end

nothing
