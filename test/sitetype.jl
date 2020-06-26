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

    function ITensors.op(::SiteType"_Custom_",
                         ::OpName"α",
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

    s = Index(4, "_Custom_")
    Sz = op("Sz", s)
    @test Sz[s'=>1,s=>1] ≈ +3/2
    @test Sz[s'=>2,s=>2] ≈ +1/2
    @test Sz[s'=>3,s=>3] ≈ -1/2
    @test Sz[s'=>4,s=>4] ≈ -3/2

    t = Index(4, "_Custom_")
    α = op("α", s, t)
    @test α[s'=>1, t'=>2, s=>1, t=>2] ≈ +3/2
    @test α[s'=>2, t'=>1, s=>2, t=>2] ≈ +1/2
    @test α[s'=>3, t'=>3, s=>3, t=>4] ≈ -1/2
    @test α[s'=>4, t'=>1, s=>4, t=>2] ≈ -3/2
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

    function ITensors.op!(::SiteType"_Custom_",
                          ::OpName"α",
                          s1::Index,
                          s2::Index)
      Op[s1'=>1, s2'=>2, s1=>1, s2=>2] = +3/2
      Op[s1'=>2, s2'=>1, s1=>2, s2=>2] = +1/2
      Op[s1'=>3, s2'=>3, s1=>3, s2=>4] = -1/2
      Op[s1'=>4, s2'=>1, s1=>4, s2=>2] = -3/2
    end

    s = Index(4, "_Custom_")
    Sz = op("Sz", s)
    @test Sz[s'=>1, s=>1] ≈ +3/2
    @test Sz[s'=>2, s=>2] ≈ +1/2
    @test Sz[s'=>3, s=>3] ≈ -1/2
    @test Sz[s'=>4, s=>4] ≈ -3/2

    t = Index(4, "_Custom_")
    α = op("α", s, t)
    @test α[s'=>1, t'=>2, s=>1, t=>2] ≈ +3/2
    @test α[s'=>2, t'=>1, s=>2, t=>2] ≈ +1/2
    @test α[s'=>3, t'=>3, s=>3, t=>4] ≈ -1/2
    @test α[s'=>4, t'=>1, s=>4, t=>2] ≈ -3/2
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

end

nothing
