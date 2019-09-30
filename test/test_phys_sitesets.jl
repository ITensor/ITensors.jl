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
    @test_throws ArgumentError state(s, "Fake")
  end

  @testset "Spin Half SiteSet" begin
    s = spinHalfSites(N)

    Sz5 = op(s,"Sz",5)
    @test hasinds(Sz5,s[5]',s[5])
     
    @test_throws ArgumentError op(s, "Fake", 2)
    @test array(op(s,"S+",3),s[3]',s[3])  ≈ [ 0.0  1.0; 0.0  0.0]
    @test array(op(s,"S-",4),s[4]',s[4])  ≈ [ 0.0  0.0; 1.0  0.0]
    @test array(op(s,"Sx",2),s[2]',s[2])  ≈ [ 0.0  0.5; 0.5  0.0]
    @test array(op(s,"iSy",2),s[2]',s[2]) ≈ [ 0.0  0.5;-0.5  0.0]
    @test array(op(s,"Sy",2),s[2]',s[2])  ≈ [0.0  -0.5im; 0.5im  0.0]
    @test array(op(s,"Sz",2),s[2]',s[2])  ≈ [ 0.5  0.0; 0.0 -0.5]
    @test array(op(s,"projUp",2),s[2]',s[2])  ≈ [ 1.0  0.0; 0.0 0.0]
    @test array(op(s,"projDn",2),s[2]',s[2])  ≈ [ 0.0  0.0; 0.0 1.0]
    @test array(op(s,"Up",2),s[2])  ≈ [1.0,0.0]
    @test array(op(s,"Dn",2),s[2])  ≈ [0.0,1.0]
  end

  @testset "Spin One Site" begin
    s = SpinOneSite(4)
    @test hastags(ind(s),"Site")
    @test hastags(ind(s),"S=1")
    @test hastags(ind(s),"n=4")

    @test val(state(s,3)) == 3
  end

  @testset "Spin One SiteSet" begin
    s = spinOneSites(N)

    Sz5 = op(s,"Sz",5)
    @test hasinds(Sz5,s[5]',s[5])
     
    @test_throws ArgumentError op(s, "Fake", 2)
    @test array(op(s,"S+",3),s[3]',s[3]) ≈ [ 0 √2 0; 0 0 √2; 0 0 0]
    @test array(op(s,"S-",3),s[3]',s[3]) ≈ [ 0 0 0; √2 0 0; 0.0 √2 0]
    @test array(op(s,"Sx",3),s[3]',s[3]) ≈ [ 0 1/√2 0; 1/√2 0 1/√2; 0 1/√2 0]
    @test array(op(s,"iSy",3),s[3]',s[3]) ≈ [ 0 1/√2 0; -1/√2 0 1/√2; 0 -1/√2 0]
    @test array(op(s,"Sy",3),s[3]',s[3]) ≈ [ 0 -1/√2im 0; +1/√2im 0 -1/√2im; 0 +1/√2im 0]
    @test array(op(s,"Sz",2),s[2]',s[2]) ≈ [1.0 0 0; 0 0 0; 0 0 -1.0]
    @test array(op(s,"Sz2",2),s[2]',s[2]) ≈ [1.0 0 0; 0 0 0; 0 0 +1.0]
    @test array(op(s,"Sx2",2),s[2]',s[2]) ≈ [0.5 0 0.5;0 1.0 0;0.5 0 0.5]
    @test array(op(s,"Sy2",2),s[2]',s[2]) ≈ [0.5 0 -0.5;0 1.0 0;-0.5 0 0.5]
    @test array(op(s,"projUp",2),s[2]',s[2]) ≈ [1.0 0 0;0 0 0;0 0 0]
    @test array(op(s,"projZ0",2),s[2]',s[2]) ≈ [0 0 0;0 1.0 0;0 0 0]
    @test array(op(s,"projDn",2),s[2]',s[2]) ≈ [0 0 0;0 0 0;0 0 1.0]
    @test array(op(s,"XUp",2),s[2]) ≈ [0.5,im*√2,0.5]
    @test array(op(s,"XZ0",2),s[2]) ≈ [im*√2,0,-im*√2]
    @test array(op(s,"XDn",2),s[2]) ≈ [0.5,-im*√2,0.5]
  end

  @testset "Electron Site" begin
    s = ElectronSite(5)
    @test hastags(ind(s),"Site")
    @test hastags(ind(s),"Electron")
    @test hastags(ind(s),"n=5")

    @test val(state(s,4)) == 4
  end

  @testset "Electron SiteSet" begin
    s = electronSites(N)

    Nup5 = op(s,"Nup",5)
    @test hasinds(Nup5,s[5]',s[5])
     
    @test_throws ArgumentError op(s, "Fake", 2)
    Nup3 = array(op(s,"Nup",3),s[3]',s[3]) 
    @test Nup3 ≈ [0. 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 1]
    Ndn3 = array(op(s,"Ndn",3),s[3]',s[3]) 
    @test Ndn3 ≈ [0. 0 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 1]
    Ntot3 = array(op(s,"Ntot",3),s[3]',s[3]) 
    @test Ntot3 ≈ [0. 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 2]
    Cup3 = array(op(s,"Cup",3),s[3]',s[3]) 
    @test Cup3 ≈ [0. 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0]
    Cdagup3 = array(op(s,"Cdagup",3),s[3]',s[3]) 
    @test Cdagup3 ≈ [0. 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 1 0]
    Cdn3 = array(op(s,"Cdn",3),s[3]',s[3]) 
    @test Cdn3 ≈ [0. 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0]
    Cdagdn3 = array(op(s,"Cdagdn",3),s[3]',s[3]) 
    @test Cdagdn3 ≈ [0. 0 0 0; 0 0 0 0; 1 0 0 0; 0 1 0 0]
    F3 = array(op(s,"F",3),s[3]',s[3]) 
    @test F3 ≈ [1. 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1]
    Fup3 = array(op(s,"Fup",3),s[3]',s[3]) 
    @test Fup3 ≈ [1. 0 0 0; 0 -1 0 0; 0 0 1 0; 0 0 0 -1]
    Fdn3 = array(op(s,"Fdn",3),s[3]',s[3]) 
    @test Fdn3 ≈ [1. 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1]
    Sz3 = array(op(s,"Sz",3),s[3]',s[3]) 
    @test Sz3 ≈ [0. 0 0 0; 0 0.5 0 0; 0 0 -0.5 0; 0 0 0 0]
    Sx3 = array(op(s,"Sx",3),s[3]',s[3]) 
    @test Sx3 ≈ [0. 0 0 0; 0 0 0.5 0; 0 0.5 0 0; 0 0 0 0]
    Sp3 = array(op(s,"S+",3),s[3]',s[3]) 
    @test Sp3 ≈ [0. 0 0 0; 0 0 1 0; 0 0 0 0; 0 0 0 0]
    Sm3 = array(op(s,"S-",3),s[3]',s[3]) 
    @test Sm3 ≈ [0. 0 0 0; 0 0 0 0; 0 1 0 0; 0 0 0 0]
    Sem = array(op(s,"Emp",3),s[3])
    @test Sem ≈ [1.0; 0.0; 0.0; 0.0]
    Sup = array(op(s,"Up",3),s[3])
    @test Sup ≈ [0.0; 1.0; 0.0; 0.0]
    Sdn = array(op(s,"Dn",3),s[3])
    @test Sdn ≈ [0.0; 0.0; 1.0; 0.0]
    Supdn = array(op(s,"UpDn",3),s[3])
    @test Supdn ≈ [0.0; 0.0; 0.0; 1.0]
  end

  @testset "tJ Site" begin
    s = tJSite(5)
    @test hastags(ind(s),"Site")
    @test hastags(ind(s),"tJ")
    @test hastags(ind(s),"n=5")

    @test val(state(s,3)) == 3
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
    Cup3 = array(op(s,"Cup",3),s[3]',s[3]) 
    @test Cup3 ≈ [0. 1 0; 0 0 0; 0 0 0]
    Cdup3 = array(op(s,"Cdagup",3),s[3]',s[3]) 
    @test Cdup3 ≈ [0 0 0; 1. 0 0; 0 0 0]
    Cdn3 = array(op(s,"Cdn",3),s[3]',s[3]) 
    @test Cdn3 ≈ [0. 0. 1; 0 0 0; 0 0 0]
    Cddn3 = array(op(s,"Cdagdn",3),s[3]',s[3]) 
    @test Cddn3 ≈ [0 0 0; 0. 0 0; 1 0 0]
    FP3 = array(op(s,"FP",3),s[3]',s[3]) 
    @test FP3 ≈ [1.0 0. 0; 0 -1.0 0; 0 0 -1.0]
    Fup3 = array(op(s,"Fup",3),s[3]',s[3]) 
    @test Fup3 ≈ [1.0 0. 0; 0 -1.0 0; 0 0 1.0]
    Fdn3 = array(op(s,"Fdn",3),s[3]',s[3]) 
    @test Fdn3 ≈ [1.0 0. 0; 0 1.0 0; 0 0 -1.0]
    Sz3 = array(op(s,"Sz",3),s[3]',s[3]) 
    @test Sz3 ≈ [0.0 0. 0; 0 0.5 0; 0 0 -0.5]
    Sx3 = array(op(s,"Sx",3),s[3]',s[3]) 
    @test Sx3 ≈ [0.0 0. 0; 0 0 1; 0 1 0]
    Sp3 = array(op(s,"Splus",3),s[3]',s[3]) 
    @test Sp3 ≈ [0.0 0. 0; 0 0 1.0; 0 0 0]
    Sm3 = array(op(s,"Sminus",3),s[3]',s[3]) 
    @test Sm3 ≈ [0.0 0. 0; 0 0 0; 0 1.0 0]
    Up3 = array(op(s,"Up",3),s[3]) 
    @test Up3 ≈ [0.0; 1.0; 0]
    Dn3 = array(op(s,"Dn",3),s[3]) 
    @test Dn3 ≈ [0.0; 0.0; 1.0]
    Em3 = array(op(s,"Emp",3),s[3]) 
    @test Em3 ≈ [1.0; 0.0; 0.0]
  end
end
