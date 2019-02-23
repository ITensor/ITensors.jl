using ITensors,
      Test

@testset "MPO Basics" begin

  N = 10
  sites = Sites(N,2)
  O = MPO(sites)
  @test length(O) == N

  O[1] = ITensor(sites[1], prime(sites[1]))
  @test hasindex(O[1],sites[1])
  @test hasindex(O[1],prime(sites[1]))

  P = copy(O)
  @test hasindex(P[1],sites[1])
  @test hasindex(P[1],prime(sites[1]))
end
