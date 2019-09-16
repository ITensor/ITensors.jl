using ITensors,
      Test

@testset "SiteSet" begin

  N = 10

  @testset "Star in operator strings" begin
    sites = spinHalfSites(N)
    @test_throws ArgumentError op(sites, "Sp", 1)
    @test sites[1] isa Index
    Sz = op(sites,"Sz",2)
    SzSz = op(sites,"Sz * Sz",2)
    @test SzSz ≈ multSiteOps(Sz,Sz)
    Sy = op(sites,"Sy",2)
    SySy = op(sites,"Sy * Sy",2)
    @test SySy ≈ multSiteOps(Sy,Sy)

    sites = spinOneSites(N)
    @test_throws ArgumentError op(sites, "Sp", 1)
    Sz = op(sites,"Sz",2)
    SzSz = op(sites,"Sz * Sz",2)
    @test SzSz ≈ multSiteOps(Sz,Sz)
    Sy = op(sites,"Sy",2)
    SySy = op(sites,"Sy * Sy",2)
    @test SySy ≈ multSiteOps(Sy,Sy)
  end
end
