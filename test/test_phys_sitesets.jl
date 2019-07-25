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

end
