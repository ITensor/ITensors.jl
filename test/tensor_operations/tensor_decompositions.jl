using ITensors
using Test

@testset "tensor_decompositons" begin
  i = Index(20, "i")
  j = Index(3, "j")
  k = Index(10, "k")
  l = Index(8, "l")
  m = Index(18, "m")

  A = randomITensor(i,j,k)
  B = randomITensor(i,k,m,l)
  C = randomITensor(i,m,l,j,k)

  # sq_factor = A * (setprime(A, 1, tags=inds(A)[1]))
  # tagname = tags(inds(A)[1])
  # D, U = eigen(sq_factor, ishermitian = true, cutoff = 1e-2)

  tucker_factors, coreA = tucker_HOSVD(A, threshold = 1e-8)
  coreA = coreA * tucker_factors[1] * tucker_factors[2] * tucker_factors[3];

  @test A ≈ coreA
  #println(norm(A-Atuck) / norm(A))

  tucker_factors, coreB = tucker_HOSVD(B, threshold = 1e-8)
  coreB = coreB * tucker_factors[1] * tucker_factors[2] * tucker_factors[3] * tucker_factors[4];

  @test B ≈ coreB

  tucker_factors, coreC = tucker_HOSVD(C, threshold = 1e-8)
  coreC = coreC * tucker_factors[1] * tucker_factors[2] * tucker_factors[3] * tucker_factors[4] * tucker_factors[5];

  @test C ≈ coreC
end
