using ITensors,
      Test

@testset "Physics SiteSets" begin

  N = 10

  @testset "Spin Half Site" begin
    s = SpinHalfSite(3)
    @test hastags(ind(s),"Site")
    @test hastags(ind(s),"S=1/2")
    @test hastags(ind(s),"n=3")

    @test val(state(s,2)) == 2
  end

  @testset "Spin Half SiteSet" begin
    sites = spinHalfSites(N)
     
    Sz_2 = op(sites,"Sz",2)
    @test Sz_2[1,1] ≈ 0.5
    @test Sz_2[2,2] ≈ -0.5

    s3 = sites[3]
    Sp_3 = op(sites,"S+",3)
    @test Sp_3[s3'(1),s3(2)] ≈ 1.0
  end

  @testset "Spin One Site" begin
    s = SpinOneSite(4)
    @test hastags(ind(s),"Site")
    @test hastags(ind(s),"S=1")
    @test hastags(ind(s),"n=4")

    @test val(state(s,3)) == 3
  end

  @testset "Spin One SiteSet" begin
    sites = spinOneSites(N)
     
    Sz_2 = op(sites,"Sz",2)
    @test Sz_2[1,1] ≈ 1.0
    @test Sz_2[2,2] ≈ 0.0
    @test Sz_2[3,3] ≈ -1.0

    s3 = sites[3]
    Sp_3 = op(sites,"S+",3)
    @test Sp_3[s3'(1),s3(2)] ≈ sqrt(2.)
    @test Sp_3[s3'(2),s3(3)] ≈ sqrt(2.)
  end

  @testset "Electron Site" begin
    s = ElectronSite(5)
    @test hastags(ind(s),"Site")
    @test hastags(ind(s),"Electron")
    @test hastags(ind(s),"n=5")

    @test val(state(s,4)) == 4
  end

  @testset "Spin One SiteSet" begin
    sites = electronSites(N)
     
    Nup_2 = op(sites,"Nup",2)
    @test Nup_2[2,2] ≈ 1.0
    @test Nup_2[4,4] ≈ 1.0
  end

  @testset "tJ Site" begin
    s = tJSite(5)
    @test hastags(ind(s),"Site")
    @test hastags(ind(s),"tJ")
    @test hastags(ind(s),"n=5")

    @test val(state(s,3)) == 3
  end

  @testset "tJ SiteSet" begin
    sites = tJSites(N)
     
    Nup_2 = op(sites,"Nup",2)
    @test Nup_2[2,2] ≈ 1.0
  end
end
