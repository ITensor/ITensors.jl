using ITensors,
      Test

@testset "Tag Types" begin

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

  @testset "Custom TagType" begin

    function ITensors.op(::TagType"S=3/2",
                         s::Index,
                         opname::AbstractString)
      Op = ITensor(s',dag(s))
      if opname == "Sz"
        Op[s'=>1,s=>1] = +3/2
        Op[s'=>2,s=>2] = +1/2
        Op[s'=>3,s=>3] = -1/2
        Op[s'=>4,s=>4] = -3/2
      end
      return Op
    end

    s = Index(4,"S=3/2")
    Sz = op(s,"Sz")
    @test Sz[s'=>1,s=>1] ≈ +3/2
    @test Sz[s'=>2,s=>2] ≈ +1/2
    @test Sz[s'=>3,s=>3] ≈ -1/2
    @test Sz[s'=>4,s=>4] ≈ -3/2
  end

end

nothing
