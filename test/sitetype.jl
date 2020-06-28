using ITensors,
      Test

@testset "SiteType" begin

  N = 10

  @testset "Star in operator strings" begin
    sites = siteinds("S=1/2",N)
    @test_throws ArgumentError op(sites, "Sp", 1)
    @test sites[1] isa Index
    Sz = op(sites,"Sz",2)
    SzSz = op(sites,"Sz * Sz",2)
    @test SzSz ≈ product(Sz, Sz)
    Sy = op(sites,"Sy",2)
    SySy = op(sites,"Sy * Sy",2)
    @test SySy ≈ product(Sy, Sy)

    sites = siteinds("S=1",N)
    @test_throws ArgumentError op(sites, "Sp", 1)
    Sz = op(sites,"Sz",2)
    SzSz = op(sites,"Sz * Sz",2)
    @test SzSz ≈ product(Sz, Sz)
    Sy = op(sites,"Sy",2)
    SySy = op(sites,"Sy * Sy",2)
    @test SySy ≈ product(Sy, Sy)
  end

  @testset "Custom SiteType using op" begin
    # Use "_Custom_" tag even though this example
    # is for S=3/2, because we might define the 
    # "S=3/2" TagType inside ITensors.jl later
    function ITensors.op(::SiteType"_Custom_",
                         ::OpName"Sz",
                         s::Index)
      Op = emptyITensor(s',dag(s))
      Op[s'=>1,s=>1] = +3/2
      Op[s'=>2,s=>2] = +1/2
      Op[s'=>3,s=>3] = -1/2
      Op[s'=>4,s=>4] = -3/2
      return Op
    end

    s = Index(4,"_Custom_")
    Sz = op("Sz",s)
    @test Sz[s'=>1,s=>1] ≈ +3/2
    @test Sz[s'=>2,s=>2] ≈ +1/2
    @test Sz[s'=>3,s=>3] ≈ -1/2
    @test Sz[s'=>4,s=>4] ≈ -3/2
  end

  @testset "Custom SiteType using op!" begin
    # Use "_Custom_" tag even though this example
    # is for S=3/2, because we might define the 
    # "S=3/2" TagType inside ITensors.jl later
    function ITensors.op!(Op::ITensor,
                          ::SiteType"_Custom_",
                          ::OpName"Sz",
                          s::Index)
      Op[s'=>1,s=>1] = +3/2
      Op[s'=>2,s=>2] = +1/2
      Op[s'=>3,s=>3] = -1/2
      Op[s'=>4,s=>4] = -3/2
    end

    s = Index(4,"_Custom_")
    Sz = op("Sz",s)
    @test Sz[s'=>1,s=>1] ≈ +3/2
    @test Sz[s'=>2,s=>2] ≈ +1/2
    @test Sz[s'=>3,s=>3] ≈ -1/2
    @test Sz[s'=>4,s=>4] ≈ -3/2
  end

  @testset "Custom SiteType using older op interface" begin
    # Use "_Custom_" tag even though this example
    # is for S=3/2, because we might define the 
    # "S=3/2" TagType inside ITensors.jl later
    function ITensors.op(::SiteType"_Custom_",
                         s::Index,
                         opname::AbstractString)
      Op = emptyITensor(s',dag(s))
      if opname=="S+"
        Op[s'(1),s(2)] = sqrt(3)
        Op[s'(2),s(3)] = 2
        Op[s'(3),s(4)] = sqrt(3)
      else
        error("Name $opname not recognized for tag \"Custom\"")
      end
      return Op
    end

    s = Index(4,"_Custom_")
    Sp = op("S+",s)
    @test Sp[s'(1),s(2)] ≈ sqrt(3)
    @test Sp[s'(2),s(3)] ≈ 2
    @test Sp[s'(3),s(4)] ≈ sqrt(3)
  end

  @testset "siteind defined by space overload" begin
    ITensors.space(::SiteType"Test1") = 4
    s = siteind("Test1",3)
    @test dim(s) == 4
    @test hastags(s,"Site,Test1,n=3")
  end

  @testset "siteind defined by siteind overload" begin
    ITensors.siteind(::SiteType"Test2",n) = Index(4,"Test2,n=$n")
    s = siteind("Test2",3)
    @test dim(s) == 4
    @test hastags(s,"Test2,n=3")
  end

  @testset "siteind defined by space overload with QN" begin
    function ITensors.space(::SiteType"Test3") 
      return [QN("T",0)=>2, QN("T",1)=>1, QN("T",2)=>1]
    end
    s = siteind("Test3",3)
    @test dim(s) == 4
    @test hasqns(s)
    @test hastags(s,"Site,Test3,n=3")
  end

  @testset "siteinds defined by space overload" begin
    function ITensors.space(::SiteType"Test4"; kwargs...) 
      conserve_qns = get(kwargs,:conserve_qns,false)
      if conserve_qns
        return [QN("T",0)=>2, QN("T",1)=>1, QN("T",2)=>1]
      end
      return 4
    end

    # Without QNs
    s = siteinds("Test4",6)
    @test length(s) == 6
    @test dim(s[1]) == 4
    for n=1:length(s)
      @test hastags(s[n],"Site,Test4,n=$n")
      @test !hasqns(s[n])
    end

    # With QNs
    s = siteinds("Test4",6;conserve_qns=true)
    @test length(s) == 6
    @test dim(s[1]) == 4
    for n=1:length(s)
      @test hastags(s[n],"Site,Test4,n=$n")
      @test hasqns(s[n])
    end

  end

  @testset "siteinds defined by siteinds overload" begin
    function ITensors.siteinds(::SiteType"Test5",N; kwargs...) 
      return [Index(4,"Test5,n=$n") for n=1:N]
    end
    s = siteinds("Test5",8)
    @test length(s) == 8
    @test dim(s[1]) == 4
    for n=1:length(s)
      @test hastags(s[n],"Test5,n=$n")
    end
  end

end

nothing
