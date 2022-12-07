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

@testset "Dense QR decomposition" begin
  n, m = 4, 8
  nm = min(n, m)
  A = randomTensor(n, m)
  Q, R = qr(A)
  @test A ≈ Q * R atol = 1e-13
  @test array(Q)' * array(Q) ≈ Diagonal(fill(1.0, nm)) atol = 1e-13
end
@testset "Dense RQ decomposition" begin
  n, m = 4, 8
  nm = min(n, m)
  A = randomTensor(n, m)
  R, Q = rq(A)
  @test A ≈ R * Q atol = 1e-13
  @test array(Q) * array(Q)' ≈ Diagonal(fill(1.0, nm)) atol = 1e-13
end

nothing
