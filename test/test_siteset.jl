using ITensors,
      Test

@testset "SiteSet" begin

  N = 10

  @testset "Star in operator strings" begin
    sites = spinHalfSites(N)
    Sz = op(sites,"Sz",2)
    SzSz = op(sites,"Sz * Sz",2)
    @test SzSz ≈ multSiteOps(Sz,Sz)
  end

end
