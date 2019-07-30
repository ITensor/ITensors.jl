using ITensors,
      LinearAlgebra, # For tr()
      Random,        # To set a seed
      Test

Random.seed!(12345)

digits(::Type{T},i,j,k) where {T} = T(i*10^2+j*10+k)

@testset "diagITensor constructors" begin
  i = Index(2,"i")
  j = Index(2,"j")

  d = diagITensor(i,j)

  @test d[i(1),j(1)] == 0.0
  @test d[i(2),j(2)] == 0.0
  @test d[i(1),j(2)] == 0.0
  @test d[i(2),j(1)] == 0.0

  d[i(1),j(1)] = 1.0
  d[i(2),j(2)] = 2.0

  @test d[i(1),j(1)] == 1.0
  @test d[i(2),j(2)] == 2.0
  @test d[i(1),j(2)] == 0.0
  @test d[i(2),j(1)] == 0.0

  @test_throws ErrorException d[i(2),j(1)] = 0.0
  @test_throws ErrorException d[i(1),j(2)] = 0.0

end

