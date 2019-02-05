using ITensors,
      ITensors.CuITensors,
      Test

@testset "cuMPS Basics" begin

  N = 10
  sites = SiteSet(N,2)
  psi = cuMPS(MPS(sites))
  @test length(psi) == N

  psi[1] = cuITensor(sites[1])
  @test hasindex(psi[1],sites[1])

  @testset "RandomCuMPS" begin
    phi = randomCuMPS(sites)
    @test hasindex(phi[1],sites[1])
    @test norm(phi[1])≈1.0
    @test hasindex(phi[4],sites[4])
    @test norm(phi[4])≈1.0
  end

end
