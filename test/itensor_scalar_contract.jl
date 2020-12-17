using Test
using ITensors
using Random

Random.seed!(1234)

@testset "Test contractions with scalar-like ITensors" begin
  i = Index(2, "i")
  j = Index(2, "j")
  k = Index(2, "k")
  α = Index(1, "α")

  is = (i, j, k)

  A = randomITensor(is..., dag(α))
  B = ITensor(2, α, α', α'')

  C = A * B
  @assert C ≈ B[1, 1, 1] * A * ITensor(1, inds(B))

  C = emptyITensor(is..., α', α'')
  C .= A .* B
  @test C ≈ B[1, 1, 1] * A * ITensor(1, inds(B))

  C = emptyITensor(shuffle([(is..., α', α'')...])...)
  C .= A .* B
  @test C ≈ B[1, 1, 1] * A * ITensor(1, inds(B))
end

