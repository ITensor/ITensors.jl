using ITensors,
      Test

@testset "Physics Sites" begin

  N = 10

  @testset "Spin Half sites" begin
    s = siteinds("S=1/2",N)

    @test state(s[1],"Up") == s[1](1)
    @test state(s[1],"Dn") == s[1](2)
    @test_throws ArgumentError state(s[1],"Fake")

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

  @testset "Spin One sites" begin
    s = siteinds("S=1",N)

    @test state(s[1],"Up") == s[1](1)
    @test state(s[1],"0")  == s[1](2)
    @test state(s[1],"Dn") == s[1](3)
    @test_throws ArgumentError state(s[1],"Fake")

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

  @testset "Fermion sites" begin
    s = siteinds("Fermion",N)

    @test state(s[1],"0")   == s[1](1)
    @test state(s[1],"1")   == s[1](2)
    @test_throws ArgumentError state(s[1],"Fake")

    N5 = op(s,"N",5)
    @test hasinds(N5,s[5]',s[5])
     
    @test_throws ArgumentError op(s, "Fake", 2)
    N3 = Array(op(s,"N",3),s[3]',s[3]) 
    @test N3 ≈ [0. 0; 0 1]
    C3 = Array(op(s,"C",3),s[3]',s[3]) 
    @test C3 ≈ [0. 1; 0 0]
    Cdag3 = Array(op(s,"Cdag",3),s[3]',s[3]) 
    @test Cdag3 ≈ [0. 0; 1 0]
    F3 = Array(op(s,"F",3),s[3]',s[3]) 
    @test F3 ≈ [1. 0; 0 -1]
    Emp = Array(op(s,"Emp",3),s[3])
    @test Emp ≈ [1.0; 0.0]
    Occ = Array(op(s,"Occ",3),s[3])
    @test Occ ≈ [0.0; 1.0]
  end

  @testset "Electron sites" begin
    s = siteinds("Electron",N)

    @test state(s[1],"0")    == s[1](1)
    @test state(s[1],"Up")   == s[1](2)
    @test state(s[1],"Dn")   == s[1](3)
    @test state(s[1],"UpDn") == s[1](4)
    @test_throws ArgumentError state(s[1],"Fake")

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

  @testset "tJ sites" begin
    s = siteinds("tJ",N)

    @test state(s[1],"0")    == s[1](1)
    @test state(s[1],"Up")   == s[1](2)
    @test state(s[1],"Dn")   == s[1](3)
    @test_throws ArgumentError state(s[1],"Fake")

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


const MySite = ITensors.TagType"MySite"

@testset "Custom Site Tag Type" begin

  function ITensors.op(::MySite,s::Index,opname::AbstractString)
    Op = ITensor(s,s')
    if opname=="MyOp"
      Op[s(1),s'(1)] = 11
      Op[s(1),s'(2)] = 12
      Op[s(2),s'(1)] = 21
      Op[s(2),s'(2)] = 22
    end
    return Op
  end

  function ITensors.state(::MySite,statename::AbstractString)
    if statename == "One"
      return 1
    elseif statename == "Two"
      return 2
    end
  end

  i = Index(2,"MySite")

  expectedOp = ITensor(i,i')
  expectedOp[i(1),i'(1)] = 11
  expectedOp[i(1),i'(2)] = 12
  expectedOp[i(2),i'(1)] = 21
  expectedOp[i(2),i'(2)] = 22
  @test norm(op(i,"MyOp")-expectedOp) < 1E-10

  @test state(i,"One") == i(1)
  @test state(i,"Two") == i(2)
end
