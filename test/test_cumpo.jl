using ITensors,
      ITensors.CuITensors,
      Test

@testset "CuMPO Basics" begin

  N = 10
  sites = SiteSet(N,2)
  O = cuMPO(sites)
  @test length(O) == N

  O[1] = cuITensor(sites[1], prime(sites[1]))
  @test hasindex(O[1],sites[1])
  @test hasindex(O[1],prime(sites[1]))

  P = copy(O)
  @test hasindex(P[1],sites[1])
  @test hasindex(P[1],prime(sites[1]))
end
