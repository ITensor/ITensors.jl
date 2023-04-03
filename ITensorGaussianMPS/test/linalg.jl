using ITensorGaussianMPS
using ITensors
using LinearAlgebra
using Test
const GMPS = ITensorGaussianMPS

@testset "Fermionic Hamiltonian diagonalization in parity-conserving frame" begin
  N = 10
  # generate random Hamiltonian in non-number-conserving space
  H = zeros(ComplexF64, 2 * N, 2 * N)
  hoffd = rand(N, N) .- 0.5 + im * (rand(N, N) .- 0.5)
  hoffd = (hoffd - transpose(hoffd)) ./ 2
  H[1:N, (N + 1):end] = hoffd
  H[(N + 1):end, 1:N] = -conj.(hoffd)
  hd = rand(N, N) .- 0.5 + im * (rand(N, N) .- 0.5)
  hd = (hd + hd') ./ 2
  H[1:N, 1:N] = -1 .* conj.(hd)
  H[(N + 1):end, (N + 1):end] = hd
  H = (H + H') ./ 2
  # compare spectrum, which can also accurately be computed via standard eigendecomposition   
  d, U = GMPS._eigen_gaussian_blocked(Hermitian(H))
  d2, _ = eigen(Hermitian(H))
  d3, _ = GMPS.eigen_gaussian(Hermitian(GMPS.interleave(H)))
  @test sort(d) ≈ sort(d2)
  @test sort(d) ≈ sort(d3)
end

@testset "Undoing arbitrary complex rotation within degenerate subspaces" begin
  A = (x -> Matrix(qr(x).Q))(randn(5, 3))
  U = (x -> Matrix(qr(x).Q))(randn(ComplexF64, 3, 3))
  AU = A * U
  B = GMPS.make_subspace_real_if_possible(AU)
  # verify that same subspace is spanned by real eigenvectors B as original eigenvectors A or AU 
  @test norm(((B * B' * A) .- A)) <= eps(Float64) * 10
  @test norm(((B * B' * AU) .- AU)) <= eps(Float64) * 10
end
