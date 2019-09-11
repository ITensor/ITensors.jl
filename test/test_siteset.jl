using ITensors,
      Test

@testset "SiteSet" begin

  N = 10

  @testset "Basic SiteSet" begin
    sites = SiteSet(N,2)

    @test length(sites) == N

    @test sites[1] isa Index
    @test dim(sites[2]) == 2
    @test state(sites,3,2) == sites[3](2)

    @test siteType(sites,1) isa Type{<:AbstractSite}
    @test siteType(sites,1) == BasicSite

    @test length(sprint(show,sites)) > 1
  end

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
    str = split(sprint(show, sites), '\n')
    @test str[1] == "SiteSet"
    @test length(str) == N + 2

    sites = spinOneSites(N)
    @test_throws ArgumentError op(sites, "Sp", 1)
    Sz = op(sites,"Sz",2)
    SzSz = op(sites,"Sz * Sz",2)
    @test SzSz ≈ multSiteOps(Sz,Sz)
    Sy = op(sites,"Sy",2)
    SySy = op(sites,"Sy * Sy",2)
    @test SySy ≈ multSiteOps(Sy,Sy)
  end
  @test length(SiteSet()) == 0
end
