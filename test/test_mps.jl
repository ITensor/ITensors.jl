using ITensors,
      Test

@testset "MPS Basics" begin

  N = 10
  sites = SiteSet(N,2)
  psi = MPS(sites)
  @test length(psi) == N

  @test siteindex(psi,2) == sites[2]
  @test hasindex(psi[3],linkindex(psi,2))
  @test hasindex(psi[3],linkindex(psi,3))

  psi[1] = ITensor(sites[1])
  @test hasindex(psi[1],sites[1])

  @testset "RandomMPS" begin
    phi = randomMPS(sites)
    @test hasindex(phi[1],sites[1])
    @test norm(phi[1])≈1.0
    @test hasindex(phi[4],sites[4])
    @test norm(phi[4])≈1.0
  end

end
