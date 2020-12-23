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
    function ITensors.op(::OpName"Sz",
                         ::SiteType"_Custom_",
                         s::Index)
      Op = emptyITensor(s',dag(s))
      Op[s'=>1,s=>1] = +3/2
      Op[s'=>2,s=>2] = +1/2
      Op[s'=>3,s=>3] = -1/2
      Op[s'=>4,s=>4] = -3/2
      return Op
    end

    function ITensors.op(::OpName"α",
                         ::SiteType"_Custom_",
                         s1::Index,
                         s2::Index)
      Op = emptyITensor(s1', s2',
                        dag(s1), dag(s2))
      Op[s1'=>1, s2'=>2, s1=>1, s2=>2] = +3/2
      Op[s1'=>2, s2'=>1, s1=>2, s2=>2] = +1/2
      Op[s1'=>3, s2'=>3, s1=>3, s2=>4] = -1/2
      Op[s1'=>4, s2'=>1, s1=>4, s2=>2] = -3/2
      return Op
    end

    function ITensors.op(::OpName"β",
                         ::SiteType"_Custom1",
                         ::SiteType"_Custom2",
                         s1::Index,
                         s2::Index)
      Op = emptyITensor(s1', s2',
                        dag(s1), dag(s2))
      Op[s1'=>1, s2'=>2, s1=>1, s2=>2] = +5/2
      Op[s1'=>2, s2'=>1, s1=>2, s2=>2] = +3/2
      Op[s1'=>3, s2'=>3, s1=>3, s2=>4] = -3/2
      Op[s1'=>4, s2'=>1, s1=>4, s2=>2] = -5/2
      return Op
    end

    s = Index(4, "_Custom_, __x")
    Sz = op("Sz", s)
    @test Sz[s'=>1,s=>1] ≈ +3/2
    @test Sz[s'=>2,s=>2] ≈ +1/2
    @test Sz[s'=>3,s=>3] ≈ -1/2
    @test Sz[s'=>4,s=>4] ≈ -3/2

    t = Index(4, "_Custom_, __x")
    α = op("α", s, t)
    @test α[s'=>1, t'=>2, s=>1, t=>2] ≈ +3/2
    @test α[s'=>2, t'=>1, s=>2, t=>2] ≈ +1/2
    @test α[s'=>3, t'=>3, s=>3, t=>4] ≈ -1/2
    @test α[s'=>4, t'=>1, s=>4, t=>2] ≈ -3/2

    s1 = Index(4, "_Custom1, __x")
    @test_throws ErrorException op("α", s, s1)

    s2 = Index(4, "_Custom2, __x")
    β = op("β", s1, s2)
    @test β[s1'=>1, s2'=>2, s1=>1, s2=>2] ≈ +5/2
    @test β[s1'=>2, s2'=>1, s1=>2, s2=>2] ≈ +3/2
    @test β[s1'=>3, s2'=>3, s1=>3, s2=>4] ≈ -3/2
    @test β[s1'=>4, s2'=>1, s1=>4, s2=>2] ≈ -5/2
    @test_throws ErrorException op("β", s2, s1)
  end

  @testset "Custom OpName with long name" begin
    function ITensors.op(::OpName"my_favorite_operator",
                         ::SiteType"S=1/2",
                         s::Index)
      Op = emptyITensor(s', dag(s))
      Op[s'=>1,s=>1] = 0.11 
      Op[s'=>1,s=>2] = 0.12
      Op[s'=>2,s=>1] = 0.21
      Op[s'=>2,s=>2] = 0.22
      return Op
    end

    s = Index(2, "S=1/2, Site")
    Sz = op("my_favorite_operator", s)
    @test Sz[s'=>1,s=>1] ≈ 0.11
    @test Sz[s'=>1,s=>2] ≈ 0.12
    @test Sz[s'=>2,s=>1] ≈ 0.21
    @test Sz[s'=>2,s=>2] ≈ 0.22
  end

  @testset "op with more than two indices" begin
    ITensors.space(::SiteType"qubit") = 2

    ITensors.op(::OpName"rand",
                ::SiteType"qubit",
                s::Index...) =
      randomITensor(prime.(s)..., dag.(s)...)

    s = siteinds("qubit", 4)
    o = op("rand", s...)
    @test norm(o) > 0
    @test order(o) == 8
    @test hassameinds(o, (prime.(s)..., s...))
  end

  @testset "Custom SiteType using op!" begin
    # Use "_Custom_" tag even though this example
    # is for S=3/2, because we might define the 
    # "S=3/2" TagType inside ITensors.jl later
    function ITensors.op!(Op::ITensor,
                          ::OpName"Sz",
                          ::SiteType"_Custom_",
                          s::Index)
      Op[s'=>1,s=>1] = +3/2
      Op[s'=>2,s=>2] = +1/2
      Op[s'=>3,s=>3] = -1/2
      Op[s'=>4,s=>4] = -3/2
    end

    function ITensors.op!(Op::ITensor,
                          ::OpName"α",
                          ::SiteType"_Custom_",
                          s1::Index,
                          s2::Index)
      Op[s1'=>1, s2'=>2, s1=>1, s2=>2] = +3/2
      Op[s1'=>2, s2'=>1, s1=>2, s2=>2] = +1/2
      Op[s1'=>3, s2'=>3, s1=>3, s2=>4] = -1/2
      Op[s1'=>4, s2'=>1, s1=>4, s2=>2] = -3/2
    end

    function ITensors.op!(Op::ITensor,
                          ::OpName"β",
                          ::SiteType"_Custom1",
                          ::SiteType"_Custom2",
                          s1::Index,
                          s2::Index)
      Op[s1'=>1, s2'=>2, s1=>1, s2=>2] = +5/2
      Op[s1'=>2, s2'=>1, s1=>2, s2=>2] = +3/2
      Op[s1'=>3, s2'=>3, s1=>3, s2=>4] = -3/2
      Op[s1'=>4, s2'=>1, s1=>4, s2=>2] = -5/2
    end

    s = Index(4, "_Custom_, __x")
    Sz = op("Sz", s)
    @test Sz[s'=>1, s=>1] ≈ +3/2
    @test Sz[s'=>2, s=>2] ≈ +1/2
    @test Sz[s'=>3, s=>3] ≈ -1/2
    @test Sz[s'=>4, s=>4] ≈ -3/2

    t = Index(4, "_Custom_, __x")
    α = op("α", s, t)
    @test α[s'=>1, t'=>2, s=>1, t=>2] ≈ +3/2
    @test α[s'=>2, t'=>1, s=>2, t=>2] ≈ +1/2
    @test α[s'=>3, t'=>3, s=>3, t=>4] ≈ -1/2
    @test α[s'=>4, t'=>1, s=>4, t=>2] ≈ -3/2

    s1 = Index(4, "_Custom1, __x")
    @test_throws ErrorException op("α", t, s1)

    s2 = Index(4, "_Custom2, __x")
    β = op("β", s1, s2)
    @test β[s1'=>1, s2'=>2, s1=>1, s2=>2] ≈ +5/2
    @test β[s1'=>2, s2'=>1, s1=>2, s2=>2] ≈ +3/2
    @test β[s1'=>3, s2'=>3, s1=>3, s2=>4] ≈ -3/2
    @test β[s1'=>4, s2'=>1, s1=>4, s2=>2] ≈ -5/2
    @test_throws ErrorException op("β", s2, s1)
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

    s = siteind("Test1")
    @test dim(s) == 4
    @test hastags(s,"Site,Test1")
  end

  @testset "siteind defined by siteind overload" begin
    ITensors.siteind(::SiteType"Test2") = Index(4,"Test2")
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
    function ITensors.space(::SiteType"Test4"; conserve_qns=false)
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

  @testset "Version of siteinds taking function argument" begin
    N = 10
    s = siteinds(n->(n==1||n==N) ? "S=1/2" : "S=1",N)
    for n in (1,N)
      @test dim(s[n]) == 2
      @test hastags(s[n],"Site,S=1/2,n=$n")
    end
    for n=2:N-1
      @test dim(s[n]) == 3
      @test hastags(s[n],"Site,S=1,n=$n")
    end
  end

  @testset "siteinds addtags keyword argument" begin
    N = 4
    s = siteinds("S=1/2",N,addtags="T")
    for n=1:N
      @test hastags(s[n],"Site,S=1/2,n=$n,T")
    end
  end

  @testset "Error for undefined tag in siteinds,space system" begin
    @test_throws ErrorException siteinds("Missing",10)
    @test_throws ErrorException siteind("Missing",3)
    @test isnothing(siteind("Missing"))
  end

end

nothing
