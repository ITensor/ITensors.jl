using ITensors,
      Test

@testset "Physics SiteSets" begin

  N = 10

  @testset "Spin Half Site" begin
    s = SpinHalfSite
    @test hastags(defaultTags(s,3),"Site")
    @test hastags(defaultTags(s,3),"S=1/2")
    @test hastags(defaultTags(s,3),"n=3")

    @test dim(s) == 2

    @test_throws ArgumentError state(s,"Fake")
  end

  @testset "Spin Half SiteSet" begin
    s = spinHalfSites(N)

    Sz5 = op(s,"Sz",5)
    @test hasinds(Sz5,s[5]',s[5])
     
    @test_throws ArgumentError op(s, "Fake", 2)
    @test Array(op(s,"S+",3),s[3]',s[3])  ≈ [ 0.0  1.0; 0.0  0.0]
    @test Array(op(s,"S-",4),s[4]',s[4])  ≈ [ 0.0  0.0; 1.0  0.0]
    @test Array(op(s,"Sx",2),s[2]',s[2])  ≈ [ 0.0  0.5; 0.5  0.0]
    @test Array(op(s,"iSy",2),s[2]',s[2]) ≈ [ 0.0  0.5;-0.5  0.0]
    @test Array(op(s,"Sy",2),s[2]',s[2])  ≈ [0.0  -0.5im; 0.5im  0.0]
    @test Array(op(s,"Sz",2),s[2]',s[2])  ≈ [ 0.5  0.0; 0.0 -0.5]
    @test Array(op(s,"projUp",2),s[2]',s[2])  ≈ [ 1.0  0.0; 0.0 0.0]
    @test Array(op(s,"projDn",2),s[2]',s[2])  ≈ [ 0.0  0.0; 0.0 1.0]
    @test Array(op(s,"Up",2),s[2])  ≈ [1.0,0.0]
    @test Array(op(s,"Dn",2),s[2])  ≈ [0.0,1.0]
  end

  @testset "Spin One Site" begin
    s = SpinOneSite
    @test hastags(defaultTags(s,4),"Site")
    @test hastags(defaultTags(s,4),"S=1")

    @test dim(s) == 3
  end

  @testset "Spin One SiteSet" begin
    s = spinOneSites(N)

    Sz5 = op(s,"Sz",5)
    @test hasinds(Sz5,s[5]',s[5])
     
    @test_throws ArgumentError op(s, "Fake", 2)
    @test Array(op(s,"S+",3),s[3]',s[3]) ≈ [ 0 √2 0; 0 0 √2; 0 0 0]
    @test Array(op(s,"S-",3),s[3]',s[3]) ≈ [ 0 0 0; √2 0 0; 0.0 √2 0]
    @test Array(op(s,"Sx",3),s[3]',s[3]) ≈ [ 0 1/√2 0; 1/√2 0 1/√2; 0 1/√2 0]
    @test Array(op(s,"iSy",3),s[3]',s[3]) ≈ [ 0 1/√2 0; -1/√2 0 1/√2; 0 -1/√2 0]
    @test Array(op(s,"Sy",3),s[3]',s[3]) ≈ [ 0 -1/√2im 0; +1/√2im 0 -1/√2im; 0 +1/√2im 0]
    @test Array(op(s,"Sz",2),s[2]',s[2]) ≈ [1.0 0 0; 0 0 0; 0 0 -1.0]
    @test Array(op(s,"Sz2",2),s[2]',s[2]) ≈ [1.0 0 0; 0 0 0; 0 0 +1.0]
    @test Array(op(s,"Sx2",2),s[2]',s[2]) ≈ [0.5 0 0.5;0 1.0 0;0.5 0 0.5]
    @test Array(op(s,"Sy2",2),s[2]',s[2]) ≈ [0.5 0 -0.5;0 1.0 0;-0.5 0 0.5]
    @test Array(op(s,"projUp",2),s[2]',s[2]) ≈ [1.0 0 0;0 0 0;0 0 0]
    @test Array(op(s,"projZ0",2),s[2]',s[2]) ≈ [0 0 0;0 1.0 0;0 0 0]
    @test Array(op(s,"projDn",2),s[2]',s[2]) ≈ [0 0 0;0 0 0;0 0 1.0]
    @test Array(op(s,"XUp",2),s[2]) ≈ [0.5,im*√2,0.5]
    @test Array(op(s,"XZ0",2),s[2]) ≈ [im*√2,0,-im*√2]
    @test Array(op(s,"XDn",2),s[2]) ≈ [0.5,-im*√2,0.5]
  end

  @testset "Electron Site" begin
    s = ElectronSite
    @test hastags(defaultTags(s,5),"Site")
    @test hastags(defaultTags(s,5),"Electron")
    @test hastags(defaultTags(s,5),"n=5")

    @test dim(s) == 4
  end

  @testset "Electron SiteSet" begin
    s = electronSites(N)

    Nup5 = op(s,"Nup",5)
    @test hasinds(Nup5,s[5]',s[5])
     
    @test_throws ArgumentError op(s, "Fake", 2)
    Nup3 = Array(op(s,"Nup",3),s[3]',s[3]) 
    @test Nup3 ≈ [0. 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 1]
    Ndn3 = Array(op(s,"Ndn",3),s[3]',s[3]) 
    @test Ndn3 ≈ [0. 0 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 1]
    Ntot3 = Array(op(s,"Ntot",3),s[3]',s[3]) 
    @test Ntot3 ≈ [0. 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 2]
    Cup3 = Array(op(s,"Cup",3),s[3]',s[3]) 
    @test Cup3 ≈ [0. 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0]
    Cdagup3 = Array(op(s,"Cdagup",3),s[3]',s[3]) 
    @test Cdagup3 ≈ [0. 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 1 0]
    Cdn3 = Array(op(s,"Cdn",3),s[3]',s[3]) 
    @test Cdn3 ≈ [0. 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0]
    Cdagdn3 = Array(op(s,"Cdagdn",3),s[3]',s[3]) 
    @test Cdagdn3 ≈ [0. 0 0 0; 0 0 0 0; 1 0 0 0; 0 1 0 0]
    F3 = Array(op(s,"F",3),s[3]',s[3]) 
    @test F3 ≈ [1. 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1]
    Fup3 = Array(op(s,"Fup",3),s[3]',s[3]) 
    @test Fup3 ≈ [1. 0 0 0; 0 -1 0 0; 0 0 1 0; 0 0 0 -1]
    Fdn3 = Array(op(s,"Fdn",3),s[3]',s[3]) 
    @test Fdn3 ≈ [1. 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1]
    Sz3 = Array(op(s,"Sz",3),s[3]',s[3]) 
    @test Sz3 ≈ [0. 0 0 0; 0 0.5 0 0; 0 0 -0.5 0; 0 0 0 0]
    Sx3 = Array(op(s,"Sx",3),s[3]',s[3]) 
    @test Sx3 ≈ [0. 0 0 0; 0 0 0.5 0; 0 0.5 0 0; 0 0 0 0]
    Sp3 = Array(op(s,"S+",3),s[3]',s[3]) 
    @test Sp3 ≈ [0. 0 0 0; 0 0 1 0; 0 0 0 0; 0 0 0 0]
    Sm3 = Array(op(s,"S-",3),s[3]',s[3]) 
    @test Sm3 ≈ [0. 0 0 0; 0 0 0 0; 0 1 0 0; 0 0 0 0]
    Sem = Array(op(s,"Emp",3),s[3])
    @test Sem ≈ [1.0; 0.0; 0.0; 0.0]
    Sup = Array(op(s,"Up",3),s[3])
    @test Sup ≈ [0.0; 1.0; 0.0; 0.0]
    Sdn = Array(op(s,"Dn",3),s[3])
    @test Sdn ≈ [0.0; 0.0; 1.0; 0.0]
    Supdn = Array(op(s,"UpDn",3),s[3])
    @test Supdn ≈ [0.0; 0.0; 0.0; 1.0]
  end

  @testset "tJ Site" begin
    s = tJSite
    @test hastags(defaultTags(s,5),"Site")
    @test hastags(defaultTags(s,5),"tJ")
    @test hastags(defaultTags(s,5),"n=5")

    @test dim(s) == 3
  end

  @testset "tJ SiteSet" begin
    s = tJSites(N)

    @test_throws ArgumentError op(s, "Fake", 2)
    Nup_2 = op(s,"Nup",2)
    @test Nup_2[2,2] ≈ 1.0
    Ndn_2 = op(s,"Ndn",2)
    @test Ndn_2[3,3] ≈ 1.0
    Ntot_2 = op(s,"Ntot",2)
    @test Ntot_2[2,2] ≈ 1.0
    @test Ntot_2[3,3] ≈ 1.0
    Cup3 = Array(op(s,"Cup",3),s[3]',s[3]) 
    @test Cup3 ≈ [0. 1 0; 0 0 0; 0 0 0]
    Cdup3 = Array(op(s,"Cdagup",3),s[3]',s[3]) 
    @test Cdup3 ≈ [0 0 0; 1. 0 0; 0 0 0]
    Cdn3 = Array(op(s,"Cdn",3),s[3]',s[3]) 
    @test Cdn3 ≈ [0. 0. 1; 0 0 0; 0 0 0]
    Cddn3 = Array(op(s,"Cdagdn",3),s[3]',s[3]) 
    @test Cddn3 ≈ [0 0 0; 0. 0 0; 1 0 0]
    FP3 = Array(op(s,"FP",3),s[3]',s[3]) 
    @test FP3 ≈ [1.0 0. 0; 0 -1.0 0; 0 0 -1.0]
    Fup3 = Array(op(s,"Fup",3),s[3]',s[3]) 
    @test Fup3 ≈ [1.0 0. 0; 0 -1.0 0; 0 0 1.0]
    Fdn3 = Array(op(s,"Fdn",3),s[3]',s[3]) 
    @test Fdn3 ≈ [1.0 0. 0; 0 1.0 0; 0 0 -1.0]
    Sz3 = Array(op(s,"Sz",3),s[3]',s[3]) 
    @test Sz3 ≈ [0.0 0. 0; 0 0.5 0; 0 0 -0.5]
    Sx3 = Array(op(s,"Sx",3),s[3]',s[3]) 
    @test Sx3 ≈ [0.0 0. 0; 0 0 1; 0 1 0]
    Sp3 = Array(op(s,"Splus",3),s[3]',s[3]) 
    @test Sp3 ≈ [0.0 0. 0; 0 0 1.0; 0 0 0]
    Sm3 = Array(op(s,"Sminus",3),s[3]',s[3]) 
    @test Sm3 ≈ [0.0 0. 0; 0 0 0; 0 1.0 0]
    Up3 = Array(op(s,"Up",3),s[3]) 
    @test Up3 ≈ [0.0; 1.0; 0]
    Dn3 = Array(op(s,"Dn",3),s[3]) 
    @test Dn3 ≈ [0.0; 0.0; 1.0]
    Em3 = Array(op(s,"Emp",3),s[3]) 
    @test Em3 ≈ [1.0; 0.0; 0.0]
  end
end
