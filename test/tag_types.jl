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

  @testset "Custom SiteType" begin

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

end

nothing
